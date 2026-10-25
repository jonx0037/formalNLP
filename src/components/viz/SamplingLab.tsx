import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/sampling-temperature-nucleus/sampling_temperature_nucleus.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_entropy_rises_with_temperature, test_temperature_divides_the_zipf_slope,
// test_local_temperature_is_not_global, test_kn_seen_set_is_recoverable_but_no_rule_finds_it,
// test_rules_share_one_frontier, test_the_text_is_off_every_curve, test_fabrication_regimes). Change the .py,
// regenerate, never the reverse.
import payload from './data/sampling-temperature-nucleus-lab.json';

/**
 * Sampling Laboratory — five figures for the sampling-temperature-nucleus topic.
 *   A. "The temperature dial": one next-token distribution (five trigram prompts, or a Zipf row) as τ slides; its
 *      entropy only rises (Theorem 2) and its log–log slope is the τ = 1 slope over τ (Proposition 2).
 *   B. "Where each rule cuts": top-k, nucleus, min-p, η, and typical on the same rows, with the kept mass, TV = 1 − m,
 *      KL = −log m (Theorem 4), and, on trigram rows, the precision and recall of the cut against the seen set
 *      (Proposition 4).
 *   C. "Local against global temperature": #14's lab tree under the per-step sampler q_τ and under π_τ ∝ p^{1/τ}
 *      (Theorem 3).
 *   D. "One frontier": every setting on both engines, model NLL against oracle NLL, with the mode and the held-out
 *      text marked.
 *   E. "Boilerplate against fabrication": the order-5 fact model's exact P(correct) and P(wrong) against τ for facts seen
 *      twice or more, once, and never.
 */

type Dial = { p: number[]; H: number; slope: number; Ez: number };
type Cut = { x: number; n: number; m: number; tv: number; kl: number; top: boolean[]; precision?: number; recall?: number };
type Rule = 'top-k' | 'nucleus' | 'min-p' | 'eta' | 'typical';
type RowData = {
  prompt: string; words: string[]; support: number; slope: number; dial: Dial[]; cuts: Record<Rule, Cut[]>;
  seen?: boolean[]; n_seen?: number;
};
const TAUS = payload.taus as number[];
const ROWS = payload.rows as unknown as RowData[];
const RULES: Rule[] = ['top-k', 'nucleus', 'min-p', 'eta', 'typical'];
const TREE = payload.tree as unknown as {
  strings: string[]; p: number[]; greedy: number; mode: number; points: { local: number[]; global: number[]; tv: number }[];
};
type Point = {
  x: number; finished: number; length: number; nll: number; oracle_nll: number; distinct2: number; distinct: number;
  copied: number; loops: number; tail: number; kl: number;
};
type Family = 'temperature' | Rule;
const FAMILIES: Family[] = ['temperature', ...RULES];
type Frontier = Record<Family, Point[]> & { heldout: { nll: number; oracle_nll: number }; mode?: { nll: number; oracle_nll: number } };
const FRONTIER = payload.frontier as unknown as Record<'trigram' | 'decoder', Frontier>;
type Regime = 'seen' | 'once' | 'unseen';
const REGIMES: Regime[] = ['seen', 'once', 'unseen'];
type FabCurve = { correct: number[]; wrong: number[] };
const FAB = payload.fabrication as unknown as Record<Regime, { temperature: FabCurve; argmax_right: number }> & {
  n: Record<Regime, number>;
  examples: { entity: string; regime: Regime; count: number; answer: string; answers: string[]; p: number[]; correct: number[] }[];
};
const ONE = TAUS.indexOf(1);

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3);
const tauText = (t: number) => (t >= 1 ? t.toFixed(2) : t.toFixed(3));

const RULE_LABEL: Record<Family, string> = {
  temperature: 'temperature', 'top-k': 'top-k', nucleus: 'nucleus', 'min-p': 'min-p', eta: 'η', typical: 'typical',
};
const PARAM_LABEL: Record<Family, string> = {
  temperature: 'τ', 'top-k': 'k', nucleus: 'p', 'min-p': 'ρ', eta: 'η', typical: 'typical mass',
};
const FAMILY_COLOR: Record<Family, string> = {
  temperature: 'var(--color-accent)', 'top-k': 'var(--color-badge-blue-text)', nucleus: 'var(--color-badge-green-text)',
  'min-p': 'var(--color-badge-amber-text)', eta: 'var(--color-badge-red-text)', typical: 'var(--color-accent-secondary)',
};
const REGIME_COLOR: Record<Regime, string> = {
  seen: 'var(--color-badge-green-text)', once: 'var(--color-badge-amber-text)', unseen: 'var(--color-badge-red-text)',
};
const REGIME_LABEL: Record<Regime, string> = { seen: 'seen twice or more', once: 'seen once', unseen: 'never seen' };

function Pill({ active, onClick, children, data }: { active: boolean; onClick: () => void; children: string; data: Record<string, string> }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} {...data}
      style={{
        fontFamily: sans, fontSize: '0.75rem', padding: '0.2rem 0.55rem', borderRadius: '999px', cursor: 'pointer',
        border: '1px solid var(--color-border)',
        background: active ? 'var(--color-accent)' : 'var(--color-surface)',
        color: active ? 'var(--color-bg)' : 'var(--color-text)',
      }}>
      {children}
    </button>
  );
}

function Readout({ label, value, id, emphasis }: { label: string; value: string; id: string; emphasis?: boolean }) {
  return (
    <div style={{ fontFamily: sans, fontSize: '0.8rem', minWidth: 0 }}>
      <div style={{ color: 'var(--color-text-secondary)' }}>{label}</div>
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)', overflowWrap: 'anywhere' }}>
        {value}
      </div>
    </div>
  );
}

function Row({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.45rem', ...style }}>{children}</div>;
}

function Readouts({ children }: { children: ReactNode }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>{children}</div>;
}

function Note({ children }: { children: ReactNode }) {
  return <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>{children}</p>;
}

function Caption({ children }: { children: ReactNode }) {
  return <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>{children}</figcaption>;
}

function Legend({ items }: { items: { label: string; color: string; dash?: string; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="8" aria-hidden="true">
            {s.dot
              ? <circle cx="11" cy="4" r="3.2" style={{ fill: s.color }} />
              : <line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />}
          </svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

function Slider({ id, label, value, max, onChange, text }: {
  id: string; label: string; value: number; max: number; onChange: (v: number) => void; text: string;
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', marginBottom: '0.45rem' }}>
      <label htmlFor={`st-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`st-lab-${id}`} type="range" min={0} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

function RowPicker({ value, onChange, name }: { value: number; onChange: (i: number) => void; name: string }) {
  return (
    <Row>
      {ROWS.map((r, i) => (
        <Pill key={r.prompt} active={value === i} onClick={() => onChange(i)} data={{ [`data-${name}`]: String(i) }}>
          {r.seen ? `${r.prompt} …` : r.prompt}
        </Pill>
      ))}
    </Row>
  );
}

// ----- Bars: the twelve likeliest tokens of a row -----

const BW = 600, BH = 230, BM = { top: 24, right: 10, bottom: 52, left: 40 };
const BIW = BW - BM.left - BM.right, BIH = BH - BM.top - BM.bottom;

function Bars({ id, words, p, kept, seen, title, aria }: {
  id: string; words: string[]; p: number[]; kept?: boolean[]; seen?: boolean[]; title: string; aria: string;
}) {
  const x = d3.scaleBand<number>().domain(d3.range(words.length)).range([0, BIW]).padding(0.18);
  const y = d3.scaleLinear().domain([0, 1]).range([BIH, 0]);
  return (
    <svg viewBox={`0 0 ${BW} ${BH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id} aria-label={aria}>
      <text x={8} y={14} style={titleText}>{title}</text>
      <g transform={`translate(${BM.left},${BM.top})`}>
        {y.ticks(4).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={BIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-5} y={y(t) + 3} textAnchor="end" style={axisText}>{String(Number(t.toFixed(2)))}</text>
          </g>
        ))}
        {words.map((w, i) => {
          const on = kept ? kept[i] : true;
          return (
            <g key={`b${i}`} data-bar={w} data-kept={kept ? String(on) : undefined}>
              <rect x={x(i)} y={y(p[i])} width={x.bandwidth()} height={BIH - y(p[i])}
                style={{ fill: on ? 'var(--color-accent)' : 'var(--color-border)', opacity: on ? 0.9 : 0.7 }} />
              {seen && seen[i] && <circle cx={x(i)! + x.bandwidth() / 2} cy={BIH + 6} r={2.6} style={{ fill: 'var(--color-badge-green-text)' }} />}
              <text transform={`translate(${x(i)! + x.bandwidth() / 2},${BIH + 13}) rotate(-35)`} textAnchor="end"
                style={{ fontFamily: mono, fontSize: '9.5px', fill: on ? 'var(--color-text)' : 'var(--color-text-secondary)' }}>
                {w}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

// ----- Figure C: the lab tree's strings under two samplers -----

function TreeBars({ ti }: { ti: number }) {
  const pt = TREE.points[ti];
  const W = 600, rowH = 19, M = { top: 26, left: 168, right: 50 };
  const H = M.top + rowH * TREE.strings.length + 8;
  const x = d3.scaleLinear().domain([0, 1]).range([0, W - M.left - M.right]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="tree"
      aria-label={`Lab tree at τ = ${tauText(TAUS[ti])}: total variation ${f3(pt.tv)} between the per-step and the global sampler`}>
      <text x={8} y={14} style={titleText}>after “revenue”: each finished string’s probability under the two samplers</text>
      {TREE.strings.map((s, i) => {
        const yy = M.top + i * rowH;
        const tag = i === TREE.greedy ? ' (greedy)' : i === TREE.mode ? ' (mode)' : '';
        return (
          <g key={s} data-string={s}>
            <text x={M.left - 6} y={yy + 11} textAnchor="end" style={{ fontFamily: mono, fontSize: '9.5px', fill: tag ? 'var(--color-accent)' : 'var(--color-text)' }}>
              {s}{tag}
            </text>
            <rect x={M.left} y={yy + 1} width={x(pt.local[i])} height={7} style={{ fill: 'var(--color-accent)' }} />
            <rect x={M.left} y={yy + 9} width={x(pt.global[i])} height={7} style={{ fill: 'var(--color-badge-blue-text)' }} />
            <text x={M.left + x(Math.max(pt.local[i], pt.global[i])) + 4} y={yy + 12} style={{ ...axisText, fontFamily: mono, fontSize: '8.5px' }}>
              {pt.local[i].toFixed(3)} / {pt.global[i].toFixed(3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ----- Figure D: the frontier -----

const FW = 600, FH = 330, FM = { top: 24, right: 16, bottom: 36, left: 46 };
const FIW = FW - FM.left - FM.right, FIH = FH - FM.top - FM.bottom;

function FrontierChart({ engine, family, idx }: { engine: 'trigram' | 'decoder'; family: Family; idx: number }) {
  const F = FRONTIER[engine];
  // The axes cover every truncation setting, τ ≤ 1.15, and the marks; hotter settings run off to the upper right.
  const core = FAMILIES.flatMap((f) => F[f].filter((p) => f !== 'temperature' || p.x <= 1.15));
  const marks = [...(engine === 'trigram' ? [F.heldout] : []), ...(F.mode ? [F.mode] : [])];
  const xs = [...core.map((p) => p.nll), ...marks.map((m) => m.nll)];
  const ys = [...core.map((p) => p.oracle_nll), ...marks.map((m) => m.oracle_nll)];
  const x = d3.scaleLinear().domain([Math.floor(d3.min(xs)! * 2) / 2, Math.ceil(d3.max(xs)! * 2) / 2]).range([0, FIW]);
  const y = d3.scaleLinear().domain([Math.floor(d3.min(ys)! * 2) / 2, Math.ceil(d3.max(ys)! * 2) / 2]).range([FIH, 0]);
  const line = d3.line<Point>().x((p) => x(p.nll)).y((p) => y(p.oracle_nll));
  const sel = F[family][idx];
  return (
    <svg viewBox={`0 0 ${FW} ${FH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="frontier"
      aria-label={`${engine}: ${RULE_LABEL[family]} ${PARAM_LABEL[family]} = ${sel.x}: model NLL ${f3(sel.nll)}, oracle NLL ${f3(sel.oracle_nll)}`}>
      <text x={8} y={14} style={titleText}>each setting’s samples: nats per word under the model (x) and under the oracle (y)</text>
      <defs>
        <clipPath id={`st-frontier-clip-${engine}`}><rect x={-8} y={-8} width={FIW + 16} height={FIH + 16} /></clipPath>
      </defs>
      <g transform={`translate(${FM.left},${FM.top})`}>
        {y.ticks(5).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={FIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-5} y={y(t) + 3} textAnchor="end" style={axisText}>{String(t)}</text>
          </g>
        ))}
        {x.ticks(6).map((t) => (
          <text key={`x${t}`} x={x(t)} y={FIH + 14} textAnchor="middle" style={axisText}>{String(t)}</text>
        ))}
        <g clipPath={`url(#st-frontier-clip-${engine})`}>
        {FAMILIES.map((f) => (
          <g key={f} data-series={f} style={{ opacity: f === family ? 1 : 0.55 }}>
            <path d={line(F[f]) ?? ''} style={{ fill: 'none', stroke: FAMILY_COLOR[f], strokeWidth: f === family ? 2.2 : 1.4 }} />
            {F[f].map((p) => <circle key={p.x} cx={x(p.nll)} cy={y(p.oracle_nll)} r={2.6} style={{ fill: FAMILY_COLOR[f] }} />)}
          </g>
        ))}
        <circle cx={x(sel.nll)} cy={y(sel.oracle_nll)} r={6} data-selected={`${family}:${sel.x}`}
          style={{ fill: 'none', stroke: 'var(--color-text)', strokeWidth: 1.5 }} />
        </g>
        {engine === 'trigram' && (
          <g data-mark="heldout">
            <rect x={x(F.heldout.nll) - 4.5} y={y(F.heldout.oracle_nll) - 4.5} width={9} height={9} transform={`rotate(45 ${x(F.heldout.nll)} ${y(F.heldout.oracle_nll)})`}
              style={{ fill: 'var(--color-text)' }} />
            <text x={x(F.heldout.nll)} y={y(F.heldout.oracle_nll) + 18} textAnchor="middle" style={axisText}>held-out text</text>
          </g>
        )}
        {F.mode && (
          <g data-mark="mode">
            <rect x={x(F.mode.nll) - 4} y={y(F.mode.oracle_nll) - 4} width={8} height={8} style={{ fill: 'var(--color-text-secondary)' }} />
            <text x={x(F.mode.nll) + 8} y={y(F.mode.oracle_nll) + 3} style={axisText}>exact mode (#14)</text>
          </g>
        )}
      </g>
      <text x={FM.left + FIW / 2} y={FH - 4} textAnchor="middle" style={axisText}>per-word NLL under the generating model (→ further into its tail)</text>
      <text transform={`translate(11,${FM.top + FIH / 2}) rotate(-90)`} textAnchor="middle" style={axisText}>per-word NLL under the oracle</text>
    </svg>
  );
}

// ----- Figure E: fabrication against τ -----

const EW = 600, EH = 250, EM = { top: 24, right: 14, bottom: 36, left: 40 };
const EIW = EW - EM.left - EM.right, EIH = EH - EM.top - EM.bottom;
const xTau = d3.scaleLog().domain([TAUS[0], TAUS[TAUS.length - 1]]).range([0, EIW]);

function FabChart({ ti }: { ti: number }) {
  const y = d3.scaleLinear().domain([0, 1]).range([EIH, 0]);
  const line = d3.line<number>().x((_, i) => xTau(TAUS[i])).y((v) => y(v));
  return (
    <svg viewBox={`0 0 ${EW} ${EH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="fabrication"
      aria-label={`At τ = ${tauText(TAUS[ti])}: ${REGIMES.map((g) => `${REGIME_LABEL[g]} correct ${f3(FAB[g].temperature.correct[ti])}`).join('; ')}`}>
      <text x={8} y={14} style={titleText}>P(the knowledge base’s answer), solid, and P(another answer), dashed — exact</text>
      <g transform={`translate(${EM.left},${EM.top})`}>
        {y.ticks(4).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={EIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-5} y={y(t) + 3} textAnchor="end" style={axisText}>{String(Number(t.toFixed(2)))}</text>
          </g>
        ))}
        {[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].map((t) => (
          <text key={`x${t}`} x={xTau(t)} y={EIH + 14} textAnchor="middle" style={{ ...axisText, fontWeight: t === 1 ? 700 : 400 }}>{t}</text>
        ))}
        <line x1={xTau(TAUS[ti])} x2={xTau(TAUS[ti])} y1={0} y2={EIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '2 3' }} />
        {REGIMES.map((g) => (
          <g key={g} data-series={g}>
            <path d={line(FAB[g].temperature.correct) ?? ''} style={{ fill: 'none', stroke: REGIME_COLOR[g], strokeWidth: 2 }} />
            <path d={line(FAB[g].temperature.wrong) ?? ''} style={{ fill: 'none', stroke: REGIME_COLOR[g], strokeWidth: 1.6, strokeDasharray: '5 3' }} />
          </g>
        ))}
      </g>
      <text x={EM.left + EIW / 2} y={EH - 4} textAnchor="middle" style={axisText}>temperature τ (log scale)</text>
    </svg>
  );
}

export default function SamplingLab() {
  const [dialRow, setDialRow] = useState(3);
  const [ti, setTi] = useState(ONE);
  const [cutRow, setCutRow] = useState(3);
  const [rule, setRule] = useState<Rule>('nucleus');
  const [ci, setCi] = useState(4);
  const [treeTi, setTreeTi] = useState(TAUS.indexOf(1));
  const [engine, setEngine] = useState<'trigram' | 'decoder'>('trigram');
  const [family, setFamily] = useState<Family>('temperature');
  const [fi, setFi] = useState(4);
  const [fabTi, setFabTi] = useState(ONE);

  // ----- A -----
  const R = ROWS[dialRow];
  const D = R.dial[ti];
  // ----- B -----
  const RB = ROWS[cutRow];
  const cuts = RB.cuts[rule];
  const cc = Math.min(ci, cuts.length - 1);
  const cut = cuts[cc];
  const pickRule = (r: Rule) => { setRule(r); setCi(Math.min(ci, RB.cuts[r].length - 1)); };
  // ----- D -----
  const F = FRONTIER[engine];
  const pts = F[family];
  const ff = Math.min(fi, pts.length - 1);
  const P = pts[ff];
  const pickFamily = (f: Family) => { setFamily(f); setFi(f === 'temperature' ? 4 : 0); };

  return (
    <div data-lab="sampling" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <Caption>A · The temperature dial — one next-token distribution as τ slides</Caption>
        <RowPicker value={dialRow} onChange={setDialRow} name="dial-row" />
        <Slider id="tau" label="temperature τ" value={ti} max={TAUS.length - 1} onChange={setTi} text={tauText(TAUS[ti])} />
        <Bars id="dial" words={R.words} p={D.p} title={`the twelve likeliest of ${R.support} tokens, at τ = ${tauText(TAUS[ti])}`}
          aria={`${R.prompt}: at τ = ${tauText(TAUS[ti])} the entropy is ${f3(D.H)} nats`} />
        <Readouts>
          <Readout id="entropy" label="entropy H(p_τ), nats" value={f3(D.H)} emphasis />
          <Readout id="slope" label="log–log (Zipf) slope" value={f3(D.slope)} />
          <Readout id="slope-1" label="log–log slope at τ = 1" value={f3(R.slope)} />
          <Readout id="ez" label="expected log-probability E[log p]" value={f3(D.Ez)} />
        </Readouts>
        <Note>
          Entropy only rises as τ rises (Theorem 2), and the log–log slope is always the τ = 1 slope divided by τ
          (Proposition 2): temperature rotates Topic 1’s rank–frequency line. Near τ = 0 the mass sits on the likeliest
          token; at large τ it spreads over all {R.support} tokens.
        </Note>
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <Caption>B · Where each rule cuts — kept tokens, the cost of the cut, and the seen set</Caption>
        <RowPicker value={cutRow} onChange={setCutRow} name="cut-row" />
        <Row>
          {RULES.map((r) => (
            <Pill key={r} active={rule === r} onClick={() => pickRule(r)} data={{ 'data-rule': r }}>{RULE_LABEL[r]}</Pill>
          ))}
        </Row>
        <Slider id="cut" label={PARAM_LABEL[rule]} value={cc} max={cuts.length - 1} onChange={setCi} text={String(cut.x)} />
        <Bars id="cut" words={RB.words} p={RB.dial[ONE].p} kept={cut.top} seen={RB.seen}
          title={`kept tokens in color; ${RB.seen ? 'a green dot marks a word seen after this context in training' : 'a Zipf row has no training counts'}`}
          aria={`${RULE_LABEL[rule]} ${PARAM_LABEL[rule]} = ${cut.x}: keeps ${cut.n} tokens with mass ${f3(cut.m)}`} />
        <Readouts>
          <Readout id="kept-n" label="tokens kept" value={`${cut.n} of ${RB.support}`} />
          <Readout id="kept-m" label="kept mass m" value={f3(cut.m)} emphasis />
          <Readout id="tv" label="TV = 1 − m" value={f3(cut.tv)} />
          <Readout id="kl" label="KL(p_S ‖ p) = −log m" value={f3(cut.kl)} />
          {cut.precision !== undefined && <Readout id="precision" label="precision vs seen set" value={f3(cut.precision)} />}
          {cut.recall !== undefined && <Readout id="recall" label="recall vs seen set" value={f3(cut.recall)} />}
          {RB.n_seen !== undefined && <Readout id="n-seen" label="words seen after this context" value={String(RB.n_seen)} />}
        </Readouts>
        <Note>
          Every cut costs exactly 1 − m in total variation and −log m in KL (Theorem 4). The Kneser–Ney row knows which
          words it saw after this context (Proposition 4), but the rules see only probabilities: no single setting of any
          rule keeps exactly the seen words on every prompt.
        </Note>
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <Caption>C · Local against global temperature — #14’s lab tree</Caption>
        <Slider id="tree-tau" label="temperature τ" value={treeTi} max={TAUS.length - 1} onChange={setTreeTi} text={tauText(TAUS[treeTi])} />
        <TreeBars ti={treeTi} />
        <Legend items={[
          { label: 'per-step sampler q_τ (temper every row)', color: 'var(--color-accent)' },
          { label: 'tempered model π_τ ∝ p(y)^(1/τ)', color: 'var(--color-badge-blue-text)' },
        ]} />
        <Readouts>
          <Readout id="tree-tv" label="TV(q_τ, π_τ)" value={f3(TREE.points[treeTi].tv)} emphasis />
          <Readout id="tree-greedy" label={`q_τ on greedy’s “${TREE.strings[TREE.greedy]}”`} value={f3(TREE.points[treeTi].local[TREE.greedy])} />
          <Readout id="tree-mode" label={`π_τ on the mode “${TREE.strings[TREE.mode]}”`} value={f3(TREE.points[treeTi].global[TREE.mode])} />
        </Readouts>
        <Note>
          At τ = 1 the two coincide. Cool the dial and they part (Theorem 3): the per-step sampler collapses onto greedy’s
          string, the tempered model onto the mode.
        </Note>
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <Caption>D · One frontier — {payload.n_prompts} prompts × {payload.n_draws} draws per setting</Caption>
        <Row>
          <Pill active={engine === 'trigram'} onClick={() => setEngine('trigram')} data={{ 'data-engine': 'trigram' }}>Kneser–Ney trigram</Pill>
          <Pill active={engine === 'decoder'} onClick={() => setEngine('decoder')} data={{ 'data-engine': 'decoder' }}>toy transformer</Pill>
        </Row>
        <Row>
          {FAMILIES.map((f) => (
            <Pill key={f} active={family === f} onClick={() => pickFamily(f)} data={{ 'data-family': f }}>{RULE_LABEL[f]}</Pill>
          ))}
        </Row>
        <Slider id="setting" label={PARAM_LABEL[family]} value={ff} max={pts.length - 1} onChange={setFi} text={String(P.x)} />
        <FrontierChart engine={engine} family={family} idx={ff} />
        <Legend items={FAMILIES.map((f) => ({ label: RULE_LABEL[f], color: FAMILY_COLOR[f] }))} />
        <Readouts>
          <Readout id="f-nll" label="NLL under the model" value={f3(P.nll)} />
          <Readout id="f-oracle" label="NLL under the oracle" value={f3(P.oracle_nll)} emphasis />
          <Readout id="f-distinct" label="distinct-2" value={f3(P.distinct2)} />
          <Readout id="f-copied" label="a training sentence verbatim" value={f3(P.copied)} />
          <Readout id="f-kl" label="KL(q ‖ p) per word" value={f3(P.kl)} />
          <Readout id="f-length" label="mean words" value={f3(P.length)} />
        </Readouts>
        <Note>
          {family === 'temperature' && P.x > 1.15 && 'This setting lies off the chart, up and to the right. '}
          {engine === 'trigram'
            ? `Every rule lands on nearly the same curve: how much it cuts matters more than how. The held-out text (${f3(F.heldout.nll)} under the model, ${f3(F.heldout.oracle_nll)} under the oracle) sits below all of them — the gap is the model’s error, which no decoding rule removes.`
            : 'The transformer’s settings also share one curve. Its held-out reference is omitted: the sentences it can score at all are a templated subset of the text.'}
        </Note>
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <Caption>E · Boilerplate against fabrication — what lowering τ does to a fact the model half knows</Caption>
        <Slider id="fab-tau" label="temperature τ" value={fabTi} max={TAUS.length - 1} onChange={setFabTi} text={tauText(TAUS[fabTi])} />
        <FabChart ti={fabTi} />
        <Legend items={REGIMES.map((g) => ({ label: `${REGIME_LABEL[g]} (${FAB.n[g]} probes)`, color: REGIME_COLOR[g] }))} />
        <Readouts>
          {REGIMES.map((g) => (
            <Readout key={g} id={`fab-${g}`} label={`${REGIME_LABEL[g]}: correct / wrong`}
              value={`${f3(FAB[g].temperature.correct[fabTi])} / ${f3(FAB[g].temperature.wrong[fabTi])}`} emphasis={g === 'once'} />
          ))}
        </Readouts>
        <div data-panel="fact-examples" style={{ fontFamily: sans, fontSize: '0.78rem', marginTop: '0.75rem' }}>
          {FAB.examples.map((e) => (
            <div key={e.entity} data-example={e.regime} style={{ borderTop: '1px solid var(--color-border)', padding: '0.4rem 0' }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0 0.5rem', alignItems: 'baseline' }}>
                <span style={{ fontWeight: 600 }}>{e.entity} operates in the …</span>
                <span style={{ color: REGIME_COLOR[e.regime] }}>{REGIME_LABEL[e.regime]} ({e.count}×)</span>
                <span style={{ color: 'var(--color-text-secondary)' }}>KB: {e.answer}</span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.15rem 0.7rem', fontFamily: mono }}>
                {e.answers.map((a, i) => (
                  <span key={a} style={{ color: a === e.answer ? 'var(--color-accent)' : 'var(--color-text)' }}>{a} {e.p[i].toFixed(3)}</span>
                ))}
              </div>
              <div style={{ color: 'var(--color-text-secondary)' }}>
                P(“{e.answer}”) at τ = {tauText(TAUS[fabTi])}: <span data-readout={`ex-${e.regime}`} style={{ fontFamily: mono, color: 'var(--color-text)' }}>{f3(e.correct[fabTi])}</span>
              </div>
            </div>
          ))}
        </div>
        <Note>
          Lowering τ sharpens whatever the model believes. For facts it saw twice or more, that removes the smoothing mass
          and with it every fabrication. For a fact it saw once, the discounted count competes with the sector prior, and
          where the prior wins, a cold sampler states the wrong sector with certainty. For facts it never saw, no setting
          helps.
        </Note>
      </figure>
    </div>
  );
}
