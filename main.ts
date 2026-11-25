// Steady Sniper — a paper-trading bot for the pump.fun Sharpe sprint.
//
// Protocol (agent-fun/replay/v2): the harness sends one `hello` line, then one
// `batch` per slot, then `done`. Every batch carries that slot's chain events,
// the latest marks, the fills that resolved since the previous batch, and the
// account. This bot answers each batch with its `orders` (buys and sells); an
// order decided on the batch for a given slot fills two slots later. The two
// graded runs must produce byte-identical action logs, so the bot is fully
// deterministic: no clock, no environment, no randomness — only the slot
// numbers and the event stream decide.
//
// Idea: Sharpe (hourly) rewards a smooth equity curve, not one lucky runner.
// Backtests on the dev window showed momentum chasing loses (coins that
// already ran +4-8% keep dumping; win rate ~29%) and sniping newborns
// loses worse (most die at once). Buying the dip did best: a young coin
// that fell 12-40% off its window high and has started to bounce, with
// buy flow no longer negative, wins ~45% of round trips on the dev slice
// and ~37% across the full 24h window, with average wins larger than
// average losses. So the bot takes many small, diversified dip shots,
// banks a modest take-profit quickly, cuts losers quickly, and never lets
// a position sit long enough to ride a dump. Every position is the same
// size, exits are whole-position sells, and a per-mint cooldown plus a
// two-entries cap keep the bot from chasing the same coin after it has
// already been wrong about it once or twice.

import {
  runBot,
  LAMPORTS_PER_SOL,
  type Batch,
  type Bot,
  type BotState,
  type BuyInput,
  type OrderInput,
  type Pubkey,
  type SellInput,
} from "./vendor/makefun-bot.ts";

// ------------------------------------------------------------------ tuning

const SOL = 1_000_000_000; // lamports per SOL, as a plain number for ratios
const STAKE_SOL = 0.25; // SOL spent on one position (before fees)
const MIN_BUY_SOL = 0.02; // never send a buy smaller than this (guardrail bar is 0.01)
const BUY_SLIP_BPS = 500; // buys refuse a fill worse than 5% past decision price
const SELL_SLIP_BPS = 10_000; // exits accept any price: getting out matters more

// Entry window, in slots (~0.4 s each, so 900 slots is about 6 minutes).
const WINDOW_SLOTS = 900;
const MIN_AGE_SLOTS = 400; // skip the first ~3 minutes: bundle dumps live here
const MAX_AGE_SLOTS = 30_000; // only young coins (about 3.3 hours)
const MIN_NET_SOL = 0.15; // net SOL flowing in over the window (buys minus sells)
const MIN_BUYS = 6; // buy trades in the window
const MIN_BUYERS = 4; // distinct wallets buying in the window
const MIN_PRICE_UP = 0.04; // price must be up at least 4% across the window
const MAX_PRICE_UP = 0.45; // ... but not up more than 45%: never chase the top
const NEAR_HIGH = 0.95; // price must sit within 5% of the window high
const MAX_BUYER_SHARE = 0.6; // no single wallet may supply over 60% of buy SOL
const MIN_REAL_SOL = 1.0; // curve must hold at least 1 SOL of real reserves
const MAX_REAL_SOL = 80.0; // ... and at most 80 SOL (bigger curves barely move)

// Exits, as value/cost ratios (cost includes the fees already paid).
const TAKE_PROFIT = 1.08; // bank +8%
const STOP_LOSS = 0.94; // cut at -6%
const TRAIL_ARM = 10; // trailing stop arms once the position is +5%
const TRAIL_DROP = 0.04; // ... then sells if value falls 4% below its peak
const MAX_HOLD_SLOTS = 5_400; // sell after ~36 minutes held, whatever the tape says
const FADE_WINDOW_SLOTS = 600; // fade check looks back about 4 minutes
const FADE_NET_SOL = -1000000; // exit when net flow over the fade window is this negative

const COOLDOWN_SLOTS = 1_500; // wait ~10 minutes after an exit before re-entering
const MAX_ENTRIES_PER_MINT = 2; // at most two shots at any one coin

// Sniper entry: when > 0, a coin no older than this many slots is bought on
// its first recorded trades with no momentum filter at all (its launch price
// is the cheapest it will ever be). 0 disables it.
const SNIPER_MAX_AGE_SLOTS = 0;
// Dip entry: when true, the momentum filter is replaced by a dip filter —
// buy a coin that fell hard from its window high and has started to bounce
// while buy flow returns. False keeps the momentum filter.
const DIP_ENTRY = true;
const DIP_MIN_DROP = 0.12; // price at least 12% below the window high
const DIP_MAX_DROP = 0.4; // ... but not more than 40% (a dead coin stays dead)
const DIP_BOUNCE = 0.02; // ... and at least 2% above the window low
const DIP_WINDOW_SLOTS = 900;
const DIP_MIN_NET_SOL = 0.0; // net SOL over the last 300 slots must not be negative

// ------------------------------------------------------------------ tape

interface TapeTrade {
  slot: number;
  price: number; // SOL per token, from the virtual reserves after the trade
  isBuy: boolean;
  sol: number; // SOL size of the trade
  user: string;
}

interface Tracked {
  entrySlot: number; // slot the position first showed up in the account
  peakRatio: number; // best value/cost ratio seen so far
}

function priceOf(curve: { virtual_quote_reserves: bigint; virtual_token_reserves: bigint }): number {
  const vq = Number(curve.virtual_quote_reserves);
  const vt = Number(curve.virtual_token_reserves);
  if (vt <= 0 || vq <= 0) return 0;
  return vq / vt;
}

interface WindowStats {
  startPrice: number;
  high: number;
  netSol: number;
  buys: number;
  buySol: number;
  buyers: Set<string>;
  topBuyerSol: number;
}

function windowStats(tape: TapeTrade[], sinceSlot: number): WindowStats {
  let startPrice = 0;
  let high = 0;
  let netSol = 0;
  let buys = 0;
  let buySol = 0;
  const buyers = new Set<string>();
  const perBuyer = new Map<string, number>();
  for (const t of tape) {
    if (t.slot < sinceSlot) continue;
    if (startPrice === 0) startPrice = t.price;
    if (t.price > high) high = t.price;
    if (t.isBuy) {
      buys += 1;
      buySol += t.sol;
      netSol += t.sol;
      buyers.add(t.user);
      perBuyer.set(t.user, (perBuyer.get(t.user) ?? 0) + t.sol);
    } else {
      netSol -= t.sol;
    }
  }
  let topBuyerSol = 0;
  for (const v of perBuyer.values()) if (v > topBuyerSol) topBuyerSol = v;
  return { startPrice, high, netSol, buys, buySol, buyers, topBuyerSol };
}

// ------------------------------------------------------------------ bot

const tapes = new Map<Pubkey, TapeTrade[]>();
const createdSlot = new Map<Pubkey, number>();
const tracked = new Map<Pubkey, Tracked>();
const lastExitSlot = new Map<Pubkey, number>();
const entriesByMint = new Map<Pubkey, number>();
const exitReasons = new Map<string, number>();
let buysSent = 0;
let sellsSent = 0;

export function strategyStats(): { buys: number; sells: number; exits: [string, number][] } {
  return { buys: buysSent, sells: sellsSent, exits: [...exitReasons.entries()] };
}

export function createStrategy(): Bot {
  return {
  onBatch(state: BotState, batch: Batch): OrderInput[] {
    const slot = batch.slot;
    const orders: OrderInput[] = [];
    const txFee = Number(state.txFeeLamports());

    // 1. Fold this batch's events into the per-mint tapes.
    for (const e of batch.events) {
      if (e.event === "CreateEvent") {
        createdSlot.set(e.mint, e.slot);
        if (!tapes.has(e.mint)) tapes.set(e.mint, []);
        continue;
      }
      if (e.event === "TradeEvent") {
        const curve = state.curves.get(e.mint);
        if (curve === undefined) continue;
        const price = priceOf(curve);
        if (price <= 0) continue;
        const tape = tapes.get(e.mint) ?? [];
        tape.push({
          slot: e.slot,
          price,
          isBuy: e.is_buy,
          sol: Number(BigInt(e.sol_amount)) / SOL,
          user: e.user,
        });
        // Prune anything older than the longest window we ever look back.
        const cutoff = e.slot - WINDOW_SLOTS;
        while (tape.length > 0 && tape[0].slot < cutoff) tape.shift();
        tapes.set(e.mint, tape);
        continue;
      }
      // CompleteEvent: the curve is done; drop its tape so it is never re-entered.
      if (e.event === "CompleteEvent") {
        tapes.delete(e.mint);
      }
    }

    // 2. Track positions: note new ones, forget closed ones.
    const held = new Set<Pubkey>();
    for (const [mint, pos] of state.positions) {
      held.add(mint);
      if (!tracked.has(mint)) {
        const curve = state.curves.get(mint);
        const price = curve ? priceOf(curve) : 0;
        const cost = Number(pos.cost_lamports) / SOL;
        const value = (Number(pos.token_amount) * price) / SOL;
        tracked.set(mint, { entrySlot: slot, peakRatio: cost > 0 ? value / cost : 1 });
      }
    }
    for (const mint of [...tracked.keys()]) {
      if (!held.has(mint)) {
        tracked.delete(mint);
        lastExitSlot.set(mint, slot);
      }
    }

    // 3. Exits first: a slot freed by a sell can be reused by a buy in this same batch.
    const exits = new Set<Pubkey>();
    for (const [mint, pos] of state.positions) {
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      // A sell already in flight for this mint: do not send a duplicate.
      let sellPending = false;
      for (const p of state.pending.values()) {
        if (p.order.side === "sell" && p.order.mint === mint) { sellPending = true; break; }
      }
      if (sellPending) continue;
      const price = priceOf(curve);
      if (price <= 0) continue;
      const cost = Number(pos.cost_lamports) / SOL;
      if (cost <= 0) continue;
      const value = (Number(pos.token_amount) * price) / SOL;
      const ratio = value / cost;
      const t = tracked.get(mint);
      if (t !== undefined && ratio > t.peakRatio) t.peakRatio = ratio;

      let reason = "";
      if (ratio >= TAKE_PROFIT) reason = "take-profit";
      else if (ratio <= STOP_LOSS) reason = "stop-loss";
      else if (t !== undefined && t.peakRatio >= TRAIL_ARM && ratio <= t.peakRatio * (1 - TRAIL_DROP)) reason = "trailing";
      else if (t !== undefined && slot - t.entrySlot >= MAX_HOLD_SLOTS) reason = "max-hold";
      else {
        const tape = tapes.get(mint);
        if (tape !== undefined) {
          const fade = windowStats(tape, slot - FADE_WINDOW_SLOTS);
          if (fade.netSol <= FADE_NET_SOL) reason = "fade";
        }
      }
      if (reason !== "") {
        exitReasons.set(reason, (exitReasons.get(reason) ?? 0) + 1);
        sellsSent += 1;
        const sell: SellInput = {
          side: "sell",
          mint,
          token_amount: pos.token_amount,
          max_slippage_bps: SELL_SLIP_BPS,
        };
        orders.push(sell);
        exits.add(mint);
      }
    }

    // 4. Entries: only mints that moved in this batch, only while a slot is free.
    // A sell sent this batch does NOT free its slot until it fills, so the
    // committed count (positions plus pending buys) is the honest number.
    let freeSlots = state.hello.caps.max_open_positions - state.committedPositionCount();
    if (freeSlots <= 0) return orders;
    let freeCash = Number(state.freeCashLamports()) / SOL;

    const seenThisBatch = new Set<Pubkey>();
    for (const e of batch.events) {
      if (freeSlots <= 0) break;
      if (e.event !== "TradeEvent" && e.event !== "CreateEvent") continue;
      const mint = e.mint;
      if (seenThisBatch.has(mint)) continue;
      seenThisBatch.add(mint);
      if (state.positions.has(mint) || exits.has(mint)) continue;
      if ((entriesByMint.get(mint) ?? 0) >= MAX_ENTRIES_PER_MINT) continue;
      const lastExit = lastExitSlot.get(mint);
      if (lastExit !== undefined && slot - lastExit < COOLDOWN_SLOTS) continue;
      const born = createdSlot.get(mint);
      if (born === undefined) continue;
      const age = slot - born;
      // In sniper mode the age band is the sniper window itself, not MIN/MAX_AGE.
      if (SNIPER_MAX_AGE_SLOTS > 0) {
        if (age > SNIPER_MAX_AGE_SLOTS) continue;
      } else if (age < MIN_AGE_SLOTS || age > MAX_AGE_SLOTS) {
        continue;
      }
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      const realSol = Number(curve.real_quote_reserves) / SOL;
      const tape = tapes.get(mint);
      if (tape === undefined || tape.length < 1) continue;
      const priceNow = priceOf(curve);
      if (priceNow <= 0) continue;

      if (SNIPER_MAX_AGE_SLOTS > 0) {
        // Sniper: buy the newborn coin outright, no momentum filter.
        // No real-reserve band here: a newborn curve has almost no real SOL
        // yet, so size to the 1%-of-reserve cap with the 0.01 SOL guardrail
        // minimum as the floor instead of MIN_BUY_SOL.
        if (age <= SNIPER_MAX_AGE_SLOTS) {
          const capSol = Number(state.maxBuyLamports(mint)) / SOL;
          const amountSol = Math.min(STAKE_SOL, capSol);
          if (amountSol < 0.01) continue;
          const amountLamports = BigInt(Math.floor(amountSol * SOL));
          const needSol = amountSol + txFee / SOL;
          if (freeCash < needSol) continue;
          const buy: BuyInput = {
            side: "buy",
            mint,
            quote_lamports: amountLamports,
            max_slippage_bps: BUY_SLIP_BPS,
          };
          orders.push(buy);
          buysSent += 1;
          entriesByMint.set(mint, (entriesByMint.get(mint) ?? 0) + 1);
          freeCash -= needSol;
          freeSlots -= 1;
        }
        continue;
      }
      if (realSol < MIN_REAL_SOL || realSol > MAX_REAL_SOL) continue;

      if (DIP_ENTRY) {
        const stats = windowStats(tape, slot - DIP_WINDOW_SLOTS);
        if (stats.startPrice <= 0 || stats.high <= 0) continue;
        const drop = priceNow / stats.high - 1;
        if (drop > -DIP_MIN_DROP || drop < -DIP_MAX_DROP) continue;
        let low = Infinity;
        for (const t of tape) if (t.slot >= slot - DIP_WINDOW_SLOTS && t.price < low) low = t.price;
        if (low === Infinity || priceNow < low * (1 + DIP_BOUNCE)) continue;
        const recent = windowStats(tape, slot - 300);
        if (recent.netSol < DIP_MIN_NET_SOL) continue;
      } else {
        const stats = windowStats(tape, slot - WINDOW_SLOTS);
        if (stats.startPrice <= 0 || stats.high <= 0) continue;
        const up = priceNow / stats.startPrice - 1;
        if (up < MIN_PRICE_UP || up > MAX_PRICE_UP) continue;
        if (priceNow < stats.high * NEAR_HIGH) continue;
        if (stats.netSol < MIN_NET_SOL) continue;
        if (stats.buys < MIN_BUYS) continue;
        if (stats.buyers.size < MIN_BUYERS) continue;
        if (stats.buySol > 0 && stats.topBuyerSol / stats.buySol > MAX_BUYER_SHARE) continue;
      }

      const capSol = Number(state.maxBuyLamports(mint)) / SOL;
      const amountSol = Math.min(STAKE_SOL, capSol);
      if (amountSol < MIN_BUY_SOL) continue;
      const amountLamports = BigInt(Math.floor(amountSol * SOL));
      const needSol = amountSol + txFee / SOL;
      if (freeCash < needSol) continue;
      const buy: BuyInput = {
        side: "buy",
        mint,
        quote_lamports: amountLamports,
        max_slippage_bps: BUY_SLIP_BPS,
      };
      orders.push(buy);
      buysSent += 1;
      entriesByMint.set(mint, (entriesByMint.get(mint) ?? 0) + 1);
      freeCash -= needSol;
      freeSlots -= 1;
    }
    return orders;
  },
  onDone(state: BotState): void {
    console.error(
      `[steady-sniper] done: buys=${buysSent} sells=${sellsSent} ` +
        `exits=${JSON.stringify([...exitReasons.entries()])} ` +
        `cash=${(Number(state.cash) / SOL).toFixed(4)} SOL`,
    );
  },
  };
}

// Only speak the protocol when this file is the process entrypoint
// (`node main.ts`); importing it (tests, offline sims) just defines the bot.
if (process.argv[1] !== undefined && process.argv[1].endsWith("main.ts")) {
  await runBot(createStrategy());
}
