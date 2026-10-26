import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/sequence-labeling-hmm-crf/sequence_labeling_hmm_crf.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_trellis_example, test_prop2_finance_example, test_well_specified_world_generative_leads_then_ties,
// test_misspecified_world_discriminative_wins, test_cor2_memm_is_capped_by_the_lagged_bayes_rule,
// test_bayes_tagger_and_thm4_on_the_worlds, test_finance_claims, test_perceptron_sweep). Change the .py, regenerate,
// never the reverse.
import payload from './data/sequence-labeling-lab.json';

/**
 * Sequence Labeling Laboratory — five figures for the sequence-labeling-hmm-crf topic.
 *   A. "Trellis": one sequence of the well-specified world, 8 states × 24 positions, shaded by the marginal (sum
 *      semiring) or by the max-marginal relative to the mode (max semiring), with the Viterbi and posterior-marginal
 *      paths and the gold states (Theorems 1 and 4).
 *   B. "CRF vs. MEMM": a finance sentence completed five ways; each token's top tag under the CRF, the causal MEMM,
 *      and the HMM. The MEMM's tags before the edit never move (Proposition 2).
 *   C. "Learning curves": HMM, indicator CRF, window CRF, and averaged perceptron against the exact Bayes tagger in the
 *      well-specified and next-state worlds, in accuracy or log-loss (Corollary 1), with the MEMM ceilings (Corollary 2).
 *   D. "Finance": the five taggers at 3, 10, and 30 training documents.
 *   E. "Perceptron": mistakes against the bound R²/γ² as the margin shrinks (Theorem 5).
 */

const TR = payload.trellis as unknown as {
  tokens: string[]; gold: number[]; viterbi: number[]; mbr: number[]; node: number[][]; maxmarg: number[][];
  mode_prob: number; K: number; correct: Record<'viterbi' | 'mbr', number>; expected: Record<'viterbi' | 'mbr', number>;
};
type ModelRow = { viterbi: number[]; top: [number, number][][] };
const P2 = payload.prop2 as unknown as {
  prefix: string[]; tags: string[];
  continuations: { words: string[]; models: Record<'crf' | 'memm_causal' | 'hmm', ModelRow> }[];
};
type Memm = { w: number[]; ceiling: number[]; mbr: number[]; viterbi: number[] };
type Curves = {
  n: number[]; bayes: number; bayes_mbr: number; bayes_nll: number;
  hmm: number[]; crf0: number[]; crf1: number[]; perceptron: number[];
  hmm_nll: number[]; crf0_nll: number[]; crf1_nll: number[]; memm: Memm;
};
const CV = payload.curves as unknown as Record<'hmm' | 'next', Curves>;
type FinModel = 'hmm' | 'memm_causal' | 'memm_full' | 'crf' | 'perceptron';
const FN = payload.finance as unknown as {
  n_docs: number[]; models: Record<FinModel, Record<'acc' | 'f1' | 'oov', number[]>>; oov_share: number[];
};
const PC = payload.perceptron as unknown as { threshold: number; gamma: number; R: number; mistakes: number; bound: number }[];

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);
const W = 640;

const COLORS = {
  hmm: 'var(--color-badge-blue-text)',
  crf0: 'var(--color-accent-secondary)',
  crf1: 'var(--color-accent)',
  crf: 'var(--color-accent)',
  perceptron: 'var(--color-badge-green-text)',
  memm_causal: 'var(--color-badge-red-text)',
  memm_full: 'var(--color-badge-amber-text)',
  bayes: 'var(--color-text)',
} as const;

function Pill({ active, onClick, children, data }: { active: boolean; onClick: () => void; children: ReactNode; data: Record<string, string> }) {
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
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.45rem', alignItems: 'center', ...style }}>{children}</div>;
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

function Label({ children }: { children: ReactNode }) {
  return <span style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>{children}</span>;
}

function Key({ items }: { items: { label: string; color: string; dash?: boolean; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{
            display: 'inline-block', width: s.dot ? '0.5rem' : '0.9rem', height: s.dash ? 0 : s.dot ? '0.5rem' : '0.2rem',
            borderTop: s.dash ? `2px dashed ${s.color}` : undefined, borderRadius: s.dot ? '50%' : undefined,
            background: s.dash ? undefined : s.dot ? 'transparent' : s.color, border: s.dot ? `1.5px solid ${s.color}` : undefined,
          }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

// ----- A. Trellis -----

function TrellisView() {
  const [ring, setRing] = useState<'sum' | 'max'>('sum');
  const [pos, setPos] = useState(8);
  const T = TR.tokens.length, K = TR.K;
  const ML = 34, MT = 26, CW = 24, CH = 24;
  const WT = ML + T * CW + 8, HT = MT + K * CH + 34;
  const grid = ring === 'sum' ? TR.node : TR.maxmarg;
  const cx = (t: number) => ML + t * CW + CW / 2;
  const cy = (k: number) => MT + k * CH + CH / 2;
  const pathD = (z: number[]) => z.map((k, t) => `${t ? 'L' : 'M'}${cx(t)},${cy(k)}`).join('');
  const col = grid[pos];
  const best = col.indexOf(Math.max(...col));
  return (
    <figure style={{ margin: 0 }}>
      <Caption>One recursion, two semirings: marginals or max-marginals on the trellis</Caption>
      <Row>
        <Label>semiring</Label>
        <Pill active={ring === 'sum'} onClick={() => setRing('sum')} data={{ 'data-ring': 'sum' }}>(+, ×): marginals</Pill>
        <Pill active={ring === 'max'} onClick={() => setRing('max')} data={{ 'data-ring': 'max' }}>(max, ×): max-marginals</Pill>
      </Row>
      <div style={{ overflowX: 'auto', maxWidth: '100%' }} data-scroll="trellis">
        <svg viewBox={`0 0 ${WT} ${HT}`} style={{ width: '100%', minWidth: '540px', height: 'auto' }} role="img" data-panel="trellis"
          aria-label={`Trellis of 8 states by 24 positions in the ${ring} semiring; Viterbi gets ${TR.correct.viterbi} of 24 tags right, posterior decoding ${TR.correct.mbr}`}>
          {d3.range(K).map((k) => (
            <text key={`s${k}`} x={ML - 6} y={cy(k) + 3} textAnchor="end" style={axisText}>{`s${k}`}</text>
          ))}
          {TR.tokens.map((tok, t) => (
            <text key={`t${t}`} x={cx(t)} y={MT - 8} textAnchor="middle"
              style={{ ...axisText, fontFamily: mono, fill: t === pos ? 'var(--color-accent)' : 'var(--color-text-secondary)' }}>{tok}</text>
          ))}
          {grid.map((c, t) => c.map((v, k) => (
            <rect key={`${t}-${k}`} x={ML + t * CW + 1} y={MT + k * CH + 1} width={CW - 2} height={CH - 2} data-cell={`${t}-${k}`}
              style={{ fill: 'var(--color-accent)', fillOpacity: 0.06 + 0.84 * v, stroke: t === pos ? 'var(--color-text)' : 'none', strokeWidth: 0.6 }} />
          )))}
          <path d={pathD(TR.mbr)} style={{ fill: 'none', stroke: 'var(--color-badge-blue-text)', strokeWidth: 2.5, opacity: 0.85 }} data-path="mbr" />
          <path d={pathD(TR.viterbi)} style={{ fill: 'none', stroke: 'var(--color-text)', strokeWidth: 1.6, strokeDasharray: '4 2' }} data-path="viterbi" />
          {TR.gold.map((k, t) => (
            <circle key={`g${t}`} cx={cx(t)} cy={cy(k)} r={5.5} style={{ fill: 'none', stroke: 'var(--color-badge-green-text)', strokeWidth: 1.6 }} />
          ))}
          {d3.range(0, T, 4).map((t) => (
            <text key={`p${t}`} x={cx(t)} y={MT + K * CH + 14} textAnchor="middle" style={axisText}>{t}</text>
          ))}
          <text x={ML + (T * CW) / 2} y={MT + K * CH + 28} textAnchor="middle" style={axisText}>position t</text>
        </svg>
      </div>
      <Key items={[{ label: 'posterior-marginal path', color: 'var(--color-badge-blue-text)' },
        { label: 'Viterbi path', color: 'var(--color-text)', dash: true },
        { label: 'gold state', color: 'var(--color-badge-green-text)', dot: true }]} />
      <Row style={{ marginTop: '0.6rem' }}>
        <Label>position</Label>
        <input type="range" min={0} max={T - 1} step={1} value={pos} onChange={(e) => setPos(Number(e.target.value))}
          aria-label="position in the sequence" data-control="position" style={{ flex: '1 1 10rem', minWidth: 0 }} />
        <span style={{ fontFamily: mono, fontSize: '0.8rem' }} data-readout="pos">t = {pos}</span>
      </Row>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.6rem', fontFamily: mono, fontSize: '0.75rem' }} data-readout="column">
        {col.map((v, k) => (
          <span key={k} style={{ color: k === best ? 'var(--color-accent)' : 'var(--color-text-secondary)', fontWeight: k === best ? 600 : 400 }}>
            s{k} {f3(v)}
          </span>
        ))}
      </div>
      <Readouts>
        <Readout label="Viterbi, tags right" value={`${TR.correct.viterbi} / 24`} id="vit-correct" />
        <Readout label="posterior decoding, tags right" value={`${TR.correct.mbr} / 24`} id="mbr-correct" emphasis />
        <Readout label="expected tags right (Viterbi / posterior)" value={`${f3(TR.expected.viterbi)} / ${f3(TR.expected.mbr)}`} id="expected" />
        <Readout label="probability of the best path" value={TR.mode_prob.toPrecision(3)} id="mode-prob" />
      </Readouts>
      <Note>
        {ring === 'sum'
          ? 'Each cell is the marginal γₜ(k), the probability that the tag at t is k. Posterior decoding takes each column\'s darkest cell (Theorem 4(b)).'
          : 'Each cell is the probability of the best path through it, divided by the best path\'s probability: exactly 1 along the Viterbi path (Theorem 1 in the max semiring).'}
        {' '}At position 8 the two semirings disagree: the marginal favors s5, the best path runs through s0.
      </Note>
    </figure>
  );
}

// ----- B. CRF vs. MEMM -----

const MODELS2 = [
  { key: 'crf', label: 'CRF' },
  { key: 'memm_causal', label: 'causal MEMM' },
  { key: 'hmm', label: 'HMM' },
] as const;

function Prop2View() {
  const [ci, setCi] = useState(1);
  const cont = P2.continuations[ci];
  const words = [...P2.prefix, ...cont.words];
  const theIdx = P2.prefix.length - 1;
  const tag = (k: number) => P2.tags[k];
  const memmThe = cont.models.memm_causal.top[theIdx][0];
  const crfThe = cont.models.crf.top[theIdx][0];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Change the words after “the”: the CRF revises, the causal MEMM cannot</Caption>
      <Row>
        <Label>continuation</Label>
        {P2.continuations.map((c, i) => (
          <Pill key={c.words.join(' ')} active={ci === i} onClick={() => setCi(i)} data={{ 'data-cont': c.words.join('-') }}>
            {c.words.join(' ')}
          </Pill>
        ))}
      </Row>
      <div role="table" aria-label="each token's most probable tag under three taggers" data-panel="prop2"
        style={{ display: 'grid', gridTemplateColumns: 'minmax(4.5rem, auto) repeat(3, minmax(0, 1fr))', gap: '2px', fontFamily: sans, fontSize: '0.75rem' }}>
        <div role="columnheader" style={{ color: 'var(--color-text-secondary)', padding: '0.2rem' }}>token</div>
        {MODELS2.map((m) => (
          <div role="columnheader" key={m.key} style={{ color: COLORS[m.key], fontWeight: 600, padding: '0.2rem' }}>{m.label}</div>
        ))}
        {words.map((w, t) => {
          const edited = t > theIdx;
          return [
            <div role="cell" key={`w${t}`} style={{ fontFamily: mono, padding: '0.2rem', fontWeight: t === theIdx ? 700 : 400,
              color: edited ? 'var(--color-accent)' : 'var(--color-text)', background: t === theIdx ? 'var(--color-muted-bg)' : undefined }}>
              {w}
            </div>,
            ...MODELS2.map((m) => {
              const [k, p] = cont.models[m.key].top[t][0];
              const vit = cont.models[m.key].viterbi[t];
              return (
                <div role="cell" key={`${m.key}${t}`} data-cell={`${m.key}-${t}`}
                  style={{ padding: '0.2rem', minWidth: 0, background: t === theIdx ? 'var(--color-muted-bg)' : undefined }}>
                  <div style={{ overflowWrap: 'anywhere' }}>{tag(k)} <span style={{ fontFamily: mono }}>{f3(p)}</span></div>
                  <div style={{ height: '3px', width: `${Math.round(p * 100)}%`, background: COLORS[m.key], opacity: 0.7 }} />
                  {vit !== k && <div style={{ fontSize: '0.68rem', color: 'var(--color-text-secondary)' }}>path: {tag(vit)}</div>}
                </div>
              );
            }),
          ];
        })}
      </div>
      <Readouts>
        <Readout label="“the” under the CRF" value={`${tag(crfThe[0])} ${f3(crfThe[1])}`} id="the-crf" emphasis />
        <Readout label="“the” under the causal MEMM" value={`${tag(memmThe[0])} ${f3(memmThe[1])}`} id="the-memm" />
      </Readouts>
      <Note>
        Each cell shows the tag with the largest marginal and its probability, and the Viterbi tag when it differs. The causal
        MEMM reads only words to the left, so by Proposition 2 nothing after “the” can move its marginals up to “the”. The
        sentence opening is a real test sentence; the continuations are edits using words seen in training.
      </Note>
    </figure>
  );
}

// ----- C. Learning curves -----

const WORLD_LABEL = { hmm: 'well-specified world', next: 'next-state world' } as const;
const SERIES = [
  { key: 'hmm', label: 'HMM' },
  { key: 'crf0', label: 'CRF, words (the HMM’s family)' },
  { key: 'crf1', label: 'CRF, ±1 word' },
  { key: 'perceptron', label: 'averaged perceptron' },
] as const;

function CurvesView() {
  const [world, setWorld] = useState<'hmm' | 'next'>('next');
  const [metric, setMetric] = useState<'acc' | 'nll'>('acc');
  const c = CV[world];
  const H = 260, M = { l: 48, r: 14, t: 14, b: 36 };
  const IW = W - M.l - M.r, IH = H - M.t - M.b;
  const series = SERIES.filter((s) => metric === 'acc' || s.key !== 'perceptron').map((s) => ({
    ...s, v: (metric === 'acc' ? c[s.key] : c[`${s.key}_nll` as 'hmm_nll']) as number[],
  }));
  const ref = metric === 'acc' ? c.bayes_mbr : c.bayes_nll;
  const all = [...series.flatMap((s) => s.v), ref];
  const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08;
  const x = d3.scaleLog().domain([10, 1000]).range([0, IW]);
  const y = d3.scaleLinear().domain([lo - pad, hi + pad]).range([IH, 0]);
  const line = d3.line<number>().x((_, i) => x(c.n[i])).y((v) => y(v));
  const last = (k: string) => series.find((s) => s.key === k)?.v[c.n.length - 1];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Generative or discriminative, where the Bayes tagger is exact</Caption>
      <Row>
        <Label>world</Label>
        {(['hmm', 'next'] as const).map((w) => (
          <Pill key={w} active={world === w} onClick={() => setWorld(w)} data={{ 'data-world': w }}>{WORLD_LABEL[w]}</Pill>
        ))}
      </Row>
      <Row>
        <Label>measure</Label>
        <Pill active={metric === 'acc'} onClick={() => setMetric('acc')} data={{ 'data-metric': 'acc' }}>token accuracy</Pill>
        <Pill active={metric === 'nll'} onClick={() => setMetric('nll')} data={{ 'data-metric': 'nll' }}>conditional log-loss</Pill>
      </Row>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="curves"
        aria-label={`${WORLD_LABEL[world]}, ${metric === 'acc' ? 'test token accuracy' : 'test log-loss'} against training sequences: ${series.map((s) => `${s.label} ${s.v.map(f3).join(', ')}`).join('; ')}; Bayes ${f3(ref)}`}>
        <g transform={`translate(${M.l},${M.t})`}>
          {y.ticks(5).map((t) => (
            <g key={t}>
              <line x1={0} x2={IW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(2)}</text>
            </g>
          ))}
          {c.n.map((n) => (
            <text key={n} x={x(n)} y={IH + 14} textAnchor="middle" style={axisText}>{n}</text>
          ))}
          <text x={IW / 2} y={IH + 30} textAnchor="middle" style={axisText}>training sequences n (log scale)</text>
          <line x1={0} x2={IW} y1={y(ref)} y2={y(ref)} style={{ stroke: COLORS.bayes, strokeWidth: 1.4, strokeDasharray: '5 3' }} data-line="bayes" />
          <text x={IW - 4} y={y(ref) - 5} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>Bayes {f3(ref)}</text>
          {series.map((s) => (
            <g key={s.key} data-series={s.key}>
              <path d={line(s.v) ?? ''} style={{ fill: 'none', stroke: COLORS[s.key], strokeWidth: 2 }} />
              {s.v.map((v, i) => <circle key={i} cx={x(c.n[i])} cy={y(v)} r={2.8} style={{ fill: COLORS[s.key] }} />)}
            </g>
          ))}
        </g>
      </svg>
      <Key items={[...series.map((s) => ({ label: s.label, color: COLORS[s.key] })),
        { label: metric === 'acc' ? 'Bayes tagger (posterior decoding)' : 'Bayes tagger', color: COLORS.bayes, dash: true }]} />
      <Readouts>
        <Readout label="HMM at n = 1000" value={f3(last('hmm')!)} id="hmm-last" />
        <Readout label="CRF, words, at n = 1000" value={f3(last('crf0')!)} id="crf0-last" />
        <Readout label="CRF, ±1 word, at n = 1000" value={f3(last('crf1')!)} id="crf1-last" emphasis />
      </Readouts>
      <div style={{ marginTop: '0.9rem', fontFamily: sans, fontSize: '0.78rem' }}>
        <div style={{ fontWeight: 600, marginBottom: '0.3rem' }}>MEMMs at n = 1000 against Corollary 2’s ceiling</div>
        <div role="table" data-panel="memm" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '2px 0.5rem' }}>
          {['lookahead', 'ceiling', 'MEMM, posterior', 'MEMM, Viterbi'].map((h) => (
            <div role="columnheader" key={h} style={{ color: 'var(--color-text-secondary)' }}>{h}</div>
          ))}
          {c.memm.w.map((w, i) => [
            <div role="cell" key={`w${w}`}>{w}</div>,
            <div role="cell" key={`c${w}`} style={{ fontFamily: mono }}>{f3(c.memm.ceiling[i])}</div>,
            <div role="cell" key={`m${w}`} style={{ fontFamily: mono }}>{f3(c.memm.mbr[i])}</div>,
            <div role="cell" key={`v${w}`} style={{ fontFamily: mono, color: c.memm.viterbi[i] > c.memm.ceiling[i] ? 'var(--color-accent)' : undefined }}>
              {f3(c.memm.viterbi[i])}
            </div>,
          ])}
        </div>
      </div>
      <Note>
        {metric === 'acc'
          ? 'Viterbi decoding of the true model scores ' + f3(c.bayes) + ' here; the dashed line is posterior decoding, the Bayes rule for token accuracy (Theorem 4). '
          : 'Log-loss is what conditional maximum likelihood optimizes: in the next-state world the CRF on words beats the HMM here even where it trails in accuracy (Corollary 1). '}
        A posterior-decoded MEMM never passes its ceiling; decoded by Viterbi it can.
      </Note>
    </figure>
  );
}

// ----- D. Finance -----

const FIN_MODELS: { key: FinModel; label: string }[] = [
  { key: 'hmm', label: 'HMM' },
  { key: 'memm_causal', label: 'causal MEMM' },
  { key: 'memm_full', label: 'MEMM, ±2 words' },
  { key: 'crf', label: 'CRF' },
  { key: 'perceptron', label: 'averaged perceptron' },
];
const FIN_METRIC = { acc: 'token accuracy', f1: 'span F1', oov: 'accuracy on unseen words' } as const;

function FinanceView() {
  const [metric, setMetric] = useState<'acc' | 'f1' | 'oov'>('f1');
  const H = 250, M = { l: 44, r: 10, t: 12, b: 36 };
  const IW = W - M.l - M.r, IH = H - M.t - M.b;
  const vals = FIN_MODELS.flatMap((m) => FN.models[m.key][metric]);
  const lo = Math.max(0, Math.floor((Math.min(...vals) - 0.05) * 10) / 10);
  const x0 = d3.scaleBand<number>().domain(FN.n_docs).range([0, IW]).padding(0.2);
  const x1 = d3.scaleBand<FinModel>().domain(FIN_MODELS.map((m) => m.key)).range([0, x0.bandwidth()]).padding(0.08);
  const y = d3.scaleLinear().domain([lo, 1]).range([IH, 0]);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>BIO tags over the finance corpus’s entity spans</Caption>
      <Row>
        <Label>measure</Label>
        {(['acc', 'f1', 'oov'] as const).map((k) => (
          <Pill key={k} active={metric === k} onClick={() => setMetric(k)} data={{ 'data-fin': k }}>{FIN_METRIC[k]}</Pill>
        ))}
      </Row>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="finance"
        aria-label={`${FIN_METRIC[metric]} at ${FN.n_docs.join(', ')} training documents: ${FIN_MODELS.map((m) => `${m.label} ${FN.models[m.key][metric].map(f3).join(', ')}`).join('; ')}`}>
        <g transform={`translate(${M.l},${M.t})`}>
          {y.ticks(5).map((t) => (
            <g key={t}>
              <line x1={0} x2={IW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(1)}</text>
            </g>
          ))}
          {FN.n_docs.map((n, i) => (
            <g key={n} transform={`translate(${x0(n)},0)`}>
              {FIN_MODELS.map((m) => {
                const v = FN.models[m.key][metric][i];
                return <rect key={m.key} x={x1(m.key)} y={y(v)} width={x1.bandwidth()} height={IH - y(v)} data-bar={`${m.key}-${n}`}
                  style={{ fill: COLORS[m.key], opacity: 0.85 }}><title>{`${m.label}, ${n} documents: ${f3(v)}`}</title></rect>;
              })}
              <text x={x0.bandwidth() / 2} y={IH + 14} textAnchor="middle" style={axisText}>{n} documents</text>
            </g>
          ))}
          <text x={IW / 2} y={IH + 30} textAnchor="middle" style={axisText}>training documents</text>
        </g>
      </svg>
      <Key items={FIN_MODELS.map((m) => ({ label: m.label, color: COLORS[m.key] }))} />
      <Readouts>
        {FN.n_docs.map((n, i) => (
          <Readout key={n} label={`${n} documents: CRF / perceptron`} id={`fin-${n}`}
            value={`${f3(FN.models.crf[metric][i])} / ${f3(FN.models.perceptron[metric][i])}`} />
        ))}
        <Readout label="share of test tokens unseen in training" value={FN.oov_share.map(f3).join(' → ')} id="oov-share" />
      </Readouts>
      <Note>
        Unseen words are almost all names: the context features carry them, which is why every tagger with word-window
        features overtakes the HMM by 30 documents. Every tagger but the HMM is fit at one prior, untuned.
      </Note>
    </figure>
  );
}

// ----- E. Perceptron -----

function PerceptronView() {
  const [i, setI] = useState(PC.length - 1);
  const H = 240, M = { l: 52, r: 14, t: 12, b: 38 };
  const IW = W - M.l - M.r, IH = H - M.t - M.b;
  const x = d3.scalePoint<number>().domain(PC.map((_, k) => k)).range([0, IW]).padding(0.5);
  const y = d3.scaleLog().domain([1, 1e5]).range([IH, 0]);
  const r = PC[i];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Mistakes against the bound R²/γ² as the margin shrinks</Caption>
      <Row>
        <Label>margin at least</Label>
        {PC.map((p, k) => (
          <Pill key={p.threshold} active={i === k} onClick={() => setI(k)} data={{ 'data-margin': `${p.threshold}` }}>{p.threshold}</Pill>
        ))}
      </Row>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="perceptron"
        aria-label={`perceptron mistakes ${PC.map((p) => p.mistakes).join(', ')} against bounds ${PC.map((p) => p.bound).join(', ')}`}>
        <g transform={`translate(${M.l},${M.t})`}>
          {[1, 10, 100, 1000, 10000, 100000].map((t) => (
            <g key={t}>
              <line x1={0} x2={IW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t >= 1000 ? `${t / 1000}k` : t}</text>
            </g>
          ))}
          {PC.map((p, k) => (
            <g key={p.threshold}>
              <line x1={x(k)} x2={x(k)} y1={y(p.mistakes)} y2={y(p.bound)} style={{ stroke: 'var(--color-border)', strokeWidth: 1 }} />
              <circle cx={x(k)} cy={y(p.bound)} r={k === i ? 5.5 : 4} style={{ fill: 'var(--color-accent-secondary)' }} data-point={`bound-${k}`} />
              <circle cx={x(k)} cy={y(p.mistakes)} r={k === i ? 5.5 : 4} style={{ fill: 'var(--color-accent)' }} data-point={`mistakes-${k}`} />
              <text x={x(k)} y={IH + 14} textAnchor="middle" style={axisText}>{f3(p.gamma)}</text>
            </g>
          ))}
          <text x={IW / 2} y={IH + 32} textAnchor="middle" style={axisText}>margin γ of the training set</text>
        </g>
      </svg>
      <Key items={[{ label: 'mistakes until an error-free pass', color: 'var(--color-accent)' },
        { label: 'bound R²/γ² (Theorem 5)', color: 'var(--color-accent-secondary)' }]} />
      <Readouts>
        <Readout label="margin γ" value={f3(r.gamma)} id="gamma" />
        <Readout label="radius R" value={f3(r.R)} id="radius" />
        <Readout label="mistakes" value={`${r.mistakes}`} id="mistakes" emphasis />
        <Readout label="bound R²/γ²" value={`${r.bound}`} id="bound" />
      </Readouts>
      <Note>
        Sixty sequences labeled by a teacher whose margin on each is at least the chosen value. The bound always holds, and it
        is loose by two to three orders of magnitude: it is a worst case over every order and every dataset with that margin.
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'trellis', label: 'Trellis' },
  { key: 'prop2', label: 'CRF vs. MEMM' },
  { key: 'curves', label: 'Learning curves' },
  { key: 'finance', label: 'Finance' },
  { key: 'perceptron', label: 'Perceptron' },
] as const;

export default function SequenceLabelingLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('trellis');
  return (
    <div data-lab="sequence-labeling" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'trellis' && <TrellisView />}
      {view === 'prop2' && <Prop2View />}
      {view === 'curves' && <CurvesView />}
      {view === 'finance' && <FinanceView />}
      {view === 'perceptron' && <PerceptronView />}
    </div>
  );
}
