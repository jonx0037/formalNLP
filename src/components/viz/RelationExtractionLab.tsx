import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/relation-extraction-distant-supervision/relation_extraction_distant_supervision.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_prop1_ds_errors_split_into_three_sources, test_cor1_ds_learns_each_templates_base_rate,
// test_cor1_slider_moves_comention_false_positives, test_cor2_one_witness_silences_the_rest,
// test_mil_trades_sentence_recall_for_precision, test_prop3_noisy_or_dominates_max, test_mil_leads_at_the_fact_level,
// test_thm3_correction_and_the_e1_estimate, test_prop4_heldout_evaluation_reports_c_times_precision,
// test_prop5_triple_recall_is_bounded_by_argument_recall, test_argument_words_let_ds_memorize_the_kb).
// Change the .py, regenerate, never the reverse.
import payload from './data/relation-extraction-lab.json';

/**
 * Relation Extraction Laboratory — five figures for the relation-extraction-distant-supervision topic.
 *   A. "DS labels": the example co-mention and three classifiers' readings of it; Proposition 1's error sources.
 *   B. "What DS learns": the co-mention templates in the (ρ, π) plane against Corollary 1's line, with a slider that
 *      removes related co-mentions from training.
 *   C. "At least one": one bag's witness weights (Theorem 2, Corollary 2), forced vs free recall (Proposition 2), and
 *      sentence vs fact scores under both aggregators (Proposition 3).
 *   D. "Incomplete KB": the keep rate c (Theorem 3, Proposition 4).
 *   E. "Pipeline & KB": gold vs predicted spans on seen and held-out names (Proposition 5), and memorization.
 */

type Mention = [number, number, string, string | null];
type Kinds = Record<'stated' | 'entailed' | 'related' | 'unrelated', number>;
type FactScore = { p: number; r: number; f1: number; ap: number };
type ModelTest = { p: number; r: number; f1: number; kinds: Kinds; facts: Record<'max' | 'noisy-or', FactScore> };
type Test = 'seen' | 'unseen';
const EX = payload.example as unknown as {
  sentence: { toks: string[]; mentions: Mention[] }; i: number; j: number; h: string; t: string; kb: string[];
  probs: Record<string, Record<string, number>>;
};
type DSLabels = { labels: number; correct: number; 'co-mention': number; entailed: number; overlap: number; precision: number;
  per_relation: Record<string, { labels: number; precision: number | null }> };
const DSL = payload.ds_labels as unknown as Record<'train' | 'seen', DSLabels>;
type Template = { types: string; order: string; between: string; right: string; n: number; rho: number; top: string; pi: number;
  fitted: number; fitted_top: number; fitted_na: number };
const TEMPLATES = payload.templates as unknown as Template[];
const KAPPA = payload.kappa as unknown as {
  kappa: number[]; rho: number[]; p: number[]; r: number[]; f1: number[]; related: number[]; unrelated: number[];
  templates: { rho: number[]; labeled: number[] }[];
};
const MODELS = payload.models as unknown as Record<string, Record<Test, ModelTest>>;
type WitnessMention = { toks: string[]; head: [number, number]; tail: [number, number]; stated: boolean; mil: number; ds: number; w: number; push: number };
const MIL = payload.mil as unknown as {
  violations: { bags: number; positive: number; violated: number; rate: number; ds_error: number };
  identifiability: Record<Test, { forced: number; free: number; n_forced: number; n_free: number }>;
  witness: { relation: string; h: string; t: string; mentions: WitnessMention[]; P_any: number };
};
const KEEP = payload.keep_grid as number[];
type PRF = { p: number; r: number; f1: number };
type Incomplete = { kb_facts: number; mass_ratio: number; e1: number; raw: PRF; by_c: PRF; by_e1: PRF;
  facts_world: number; facts_kb: number; facts_indep: number; facts_world_r: number };
const INC = payload.incomplete as unknown as Record<string, Incomplete>;
type Pipe = { p: number; r: number; f1: number; bound: number; facts: PRF; arg_miss: Record<string, number> };
const PIPE = payload.pipeline as unknown as Record<Test, Record<'gold' | 'predicted', Record<'gold' | 'ds' | 'mil', Pipe>>>;
const SIZES = payload.sizes as Record<string, { sentences: number; candidates: number; triples: number }>;

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);
const W = 640;

const SOURCE_COLORS: Record<string, string> = {
  correct: 'var(--color-accent)', 'co-mention': 'var(--color-badge-blue-text)', entailed: 'var(--color-badge-green-text)',
  overlap: 'var(--color-badge-amber-text)',
};
const MODEL_COLORS: Record<string, string> = {
  gold: 'var(--color-text-secondary)', DS: 'var(--color-badge-blue-text)', MIL: 'var(--color-accent)',
};

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

function Sub({ children }: { children: ReactNode }) {
  return <div style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.7rem 0 0.3rem' }}>{children}</div>;
}

function Key({ items }: { items: { label: string; color: string; dash?: boolean; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{
            display: 'inline-block', width: s.dot ? '0.55rem' : '0.9rem', height: s.dash ? 0 : s.dot ? '0.55rem' : '0.2rem',
            borderRadius: s.dot ? '50%' : undefined,
            borderTop: s.dash ? `2px dashed ${s.color}` : undefined, background: s.dash ? undefined : s.color,
          }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

function TestPills({ test, setTest }: { test: Test; setTest: (t: Test) => void }) {
  return (
    <Row>
      <Label>test</Label>
      <Pill active={test === 'seen'} onClick={() => setTest('seen')} data={{ 'data-test': 'seen' }}>seen companies</Pill>
      <Pill active={test === 'unseen'} onClick={() => setTest('unseen')} data={{ 'data-test': 'unseen' }}>held-out companies</Pill>
    </Row>
  );
}

/** A sentence with its head mention and tail mention marked. */
function PairLine({ toks, head, tail, id }: { toks: string[]; head: [number, number]; tail: [number, number]; id: string }) {
  return (
    <div data-pair={id} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.1rem 0.3rem', fontFamily: mono, fontSize: '0.78rem' }}>
      {toks.map((w, k) => {
        const role = head[0] <= k && k < head[1] ? 'head' : tail[0] <= k && k < tail[1] ? 'tail' : null;
        return (
          <span key={k} style={{
            padding: '0 0.15rem',
            borderBottom: role === 'head' ? '3px solid var(--color-accent)' : role === 'tail' ? '3px solid var(--color-badge-blue-text)' : '3px solid transparent',
          }}>{w}</span>
        );
      })}
    </div>
  );
}

/** Horizontal bars with labels on the left and values on the right. */
function Bars({ bars, id, marks, labelWidth = 150 }: {
  bars: { label: string; v: number; color: string; id: string; faint?: boolean }[]; id: string;
  marks?: { x: number; label: string }[]; labelWidth?: number;
}) {
  const bh = 22;
  const H = bars.length * bh + 22, MG = { l: labelWidth, r: 44, t: 4 };
  const IW = W - MG.l - MG.r;
  const x = d3.scaleLinear().domain([0, 1]).range([0, IW]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id}
      aria-label={bars.map((b) => `${b.label} ${f3(b.v)}`).join(', ')}>
      <g transform={`translate(${MG.l},${MG.t})`}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={0} y2={bh * bars.length} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={x(v)} y={bh * bars.length + 12} textAnchor="middle" style={axisText}>{v}</text>
          </g>
        ))}
        {bars.map((b, i) => (
          <g key={b.id}>
            <text x={-6} y={i * bh + bh / 2 + 3} textAnchor="end" style={axisText}>{b.label}</text>
            <rect x={0} y={i * bh + 4} width={x(b.v)} height={bh - 8} data-bar={b.id}
              style={{ fill: b.color, fillOpacity: b.faint ? 0.4 : 0.85 }} />
            <text x={x(b.v) + 4} y={i * bh + bh / 2 + 3} style={axisText}>{f3(b.v)}</text>
          </g>
        ))}
        {marks?.map((m) => (
          <g key={m.label}>
            <line x1={x(m.x)} x2={x(m.x)} y1={0} y2={bh * bars.length} data-mark={m.label}
              style={{ stroke: 'var(--color-text)', strokeWidth: 1.2, strokeDasharray: '5 3' }} />
          </g>
        ))}
      </g>
    </svg>
  );
}

type Series = { label: string; ys: number[]; color: string; dash?: boolean; id: string };

function LineChart({ xs, series, xLabel, cursor, yDomain, id }: {
  xs: number[]; series: Series[]; xLabel: string; cursor?: number; yDomain: [number, number]; id: string;
}) {
  const H = 240, M = { l: 40, r: 14, t: 12, b: 36 };
  const IW = W - M.l - M.r, IH = H - M.t - M.b;
  const x = d3.scaleLinear().domain([xs[0], xs[xs.length - 1]]).range([0, IW]);
  const y = d3.scaleLinear().domain(yDomain).range([IH, 0]);
  const line = (ys: number[]) => d3.line<number>().x((_, i) => x(xs[i])).y((v) => y(v))(ys) ?? '';
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id}
      aria-label={series.map((s) => `${s.label}: ${s.ys.map(f3).join(', ')}`).join('; ')}>
      <g transform={`translate(${M.l},${M.t})`}>
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line x1={0} x2={IW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t}</text>
          </g>
        ))}
        {xs.map((t) => (
          <text key={t} x={x(t)} y={IH + 14} textAnchor="middle" style={axisText}>{t}</text>
        ))}
        <text x={IW / 2} y={IH + 30} textAnchor="middle" style={axisText}>{xLabel}</text>
        {cursor !== undefined && (
          <line x1={x(cursor)} x2={x(cursor)} y1={0} y2={IH} style={{ stroke: 'var(--color-accent)', strokeWidth: 1.5, opacity: 0.6 }} />
        )}
        {series.map((s) => (
          <g key={s.id}>
            <path d={line(s.ys)} data-series={s.id}
              style={{ fill: 'none', stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash ? '5 3' : undefined }} />
            {s.ys.map((v, i) => <circle key={i} cx={x(xs[i])} cy={y(v)} r={2.2} style={{ fill: s.color }} />)}
          </g>
        ))}
      </g>
    </svg>
  );
}

// ----- A. DS labels -----

function LabelsView() {
  const [corpus, setCorpus] = useState<'train' | 'seen'>('train');
  const ms = EX.sentence.mentions;
  const head: [number, number] = [ms[EX.i][0], ms[EX.i][1]];
  const tail: [number, number] = [ms[EX.j][0], ms[EX.j][1]];
  const d = DSL[corpus];
  const parts = [
    { key: 'correct', v: d.correct }, { key: 'co-mention', v: d['co-mention'] },
    { key: 'entailed', v: d.entailed }, { key: 'overlap', v: d.overlap },
  ];
  const IW = W - 20;
  let acc = 0;
  const rels = Object.entries(d.per_relation);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Distant supervision labels the pair, not the sentence</Caption>
      <PairLine toks={EX.sentence.toks} head={head} tail={tail} id="example" />
      <Note>
        Head <strong>{EX.h}</strong>, tail <strong>{EX.t}</strong>. The sentence states nothing; the knowledge base holds{' '}
        <span data-readout="kb">{EX.kb.join(', ')}</span>, so distant supervision labels it {EX.kb.join(', ')}.
      </Note>
      <Sub>each classifier&rsquo;s probabilities for this candidate (labels at 0.005 or more)</Sub>
      <div data-panel="example-probs" style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        {(['gold', 'DS', 'MIL'] as const).map((m) => (
          <div key={m} data-model={m} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.2rem 0.7rem', fontFamily: sans, fontSize: '0.75rem' }}>
            <span style={{ fontWeight: 600, color: MODEL_COLORS[m], minWidth: '3.2rem' }}>{m}</span>
            {Object.entries(EX.probs[m]).filter(([, v]) => v >= 0.005).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
              <span key={k} style={{ fontFamily: mono }}>{k} {f3(v)}</span>
            ))}
          </div>
        ))}
      </div>
      <Sub>every distant label, by Proposition 1&rsquo;s source</Sub>
      <Row>
        <Label>corpus</Label>
        <Pill active={corpus === 'train'} onClick={() => setCorpus('train')} data={{ 'data-corpus': 'train' }}>training</Pill>
        <Pill active={corpus === 'seen'} onClick={() => setCorpus('seen')} data={{ 'data-corpus': 'seen' }}>seen test</Pill>
      </Row>
      <svg viewBox={`0 0 ${W} 34`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="sources"
        aria-label={parts.map((p) => `${p.key} ${p.v}`).join(', ')}>
        <g transform="translate(10,4)">
          {parts.map((p) => {
            const x0 = (acc / d.labels) * IW;
            acc += p.v;
            return <rect key={p.key} x={x0} y={0} width={(p.v / d.labels) * IW} height={24} data-source={p.key}
              style={{ fill: SOURCE_COLORS[p.key], fillOpacity: 0.85, stroke: 'var(--color-bg)', strokeWidth: 1 }} />;
          })}
        </g>
      </svg>
      <Key items={parts.map((p) => ({ label: `${p.key} ${p.v.toLocaleString('en-US')}`, color: SOURCE_COLORS[p.key] }))} />
      <Readouts>
        <Readout label="distant labels" value={d.labels.toLocaleString('en-US')} id="labels" />
        <Readout label="precision (stated / labels)" value={f3(d.precision)} id="precision" emphasis />
      </Readouts>
      <Sub>precision by relation</Sub>
      <Bars id="per-relation" labelWidth={120}
        bars={rels.map(([r, v]) => ({ label: `${r} (${v.labels})`, v: v.precision ?? 0, color: 'var(--color-accent)', id: r }))} />
      <Note>
        Co-mentions state nothing; entailed labels follow from what the sentence states by the knowledge base&rsquo;s rules
        (BASED_IN from &ldquo;headquartered in Toronto, Canada&rdquo;) and count as errors only because the annotation
        guideline marks them unstated; overlaps are a second relation of the same pair (ACQUIRED beside PARENT_OF).
      </Note>
    </figure>
  );
}

// ----- B. What DS learns -----

function LearnsView() {
  const [k, setK] = useState(KAPPA.kappa.length - 1);
  const [hover, setHover] = useState<number | null>(null);
  const H = 300, M = { l: 44, r: 14, t: 12, b: 38 };
  const IW = W - M.l - M.r, IH = H - M.t - M.b;
  const x = d3.scaleLinear().domain([0, 1]).range([0, IW]);
  const y = d3.scaleLinear().domain([0, 1.05]).range([IH, 0]);
  const rhos = d3.range(0.5, 1.0001, 0.01);
  const curve = d3.line<number>().x((r) => x(r)).y((r) => y((1 - r) / r))(rhos) ?? '';
  const sel = hover ?? 1;
  const t = TEMPLATES[sel];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>A co-mention template&rsquo;s label is its base rate: argmax picks a relation iff ρπ &gt; 1 − ρ</Caption>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="templates"
        aria-label={TEMPLATES.map((tp, i) => `${tp.between}: rho ${f3(KAPPA.templates[i].rho[k])}, pi ${f3(tp.pi)}, labeled ${f3(KAPPA.templates[i].labeled[k])}`).join('; ')}>
        <g transform={`translate(${M.l},${M.t})`}>
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={0} x2={IW} y1={y(v)} y2={y(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={-6} y={y(v) + 3} textAnchor="end" style={axisText}>{v}</text>
              <text x={x(v)} y={IH + 14} textAnchor="middle" style={axisText}>{v}</text>
            </g>
          ))}
          <text x={IW / 2} y={IH + 32} textAnchor="middle" style={axisText}>ρ: share of the template&rsquo;s training pairs the KB relates</text>
          <text x={-34} y={IH / 2} transform={`rotate(-90 -34 ${IH / 2})`} textAnchor="middle" style={axisText}>π: top relation&rsquo;s share (κ = 1)</text>
          <path d={curve} data-curve="threshold" style={{ fill: 'none', stroke: 'var(--color-text)', strokeWidth: 1.2, strokeDasharray: '5 3' }} />
          <text x={x(0.56)} y={y(1.0)} style={axisText}>ρπ = 1 − ρ</text>
          {TEMPLATES.map((tp, i) => {
            const labeled = KAPPA.templates[i].labeled[k] > 0.5;
            return (
              <circle key={i} cx={x(KAPPA.templates[i].rho[k])} cy={y(tp.pi)} r={sel === i ? 7 : 5} data-template={i}
                onMouseEnter={() => setHover(i)} onClick={() => setHover(i)}
                style={{
                  fill: labeled ? 'var(--color-badge-blue-text)' : 'var(--color-surface)', cursor: 'pointer',
                  stroke: labeled ? 'var(--color-badge-blue-text)' : 'var(--color-text-secondary)', strokeWidth: 1.5,
                }} />
            );
          })}
        </g>
      </svg>
      <Key items={[
        { label: 'DS labels the template with a relation (majority of its test candidates)', color: 'var(--color-badge-blue-text)', dot: true },
        { label: 'DS labels it NA', color: 'var(--color-text-secondary)', dot: true },
        { label: 'Corollary 1’s threshold', color: 'var(--color-text)', dash: true },
      ]} />
      <Row style={{ marginTop: '0.6rem' }}>
        <Label>κ</Label>
        <input type="range" min={0} max={KAPPA.kappa.length - 1} step={1} value={k} onChange={(e) => setK(Number(e.target.value))}
          aria-label="share of related co-mentions kept in training" data-control="kappa" style={{ flex: '1 1 10rem', minWidth: 0 }} />
        <span style={{ fontFamily: mono, fontSize: '0.8rem' }} data-readout="kappa">κ = {KAPPA.kappa[k].toFixed(1)}</span>
      </Row>
      <Readouts>
        <Readout label="DS sentence F1 (seen test)" value={f3(KAPPA.f1[k])} id="kappa-f1" emphasis />
        <Readout label="related co-mentions labeled" value={f3(KAPPA.related[k])} id="kappa-related" />
        <Readout label="P / R" value={`${f3(KAPPA.p[k])} / ${f3(KAPPA.r[k])}`} id="kappa-pr" />
      </Readouts>
      <div data-panel="template-detail" style={{ fontFamily: sans, fontSize: '0.78rem', marginTop: '0.7rem', borderTop: '1px solid var(--color-border)', paddingTop: '0.5rem' }}>
        <div style={{ fontFamily: mono, overflowWrap: 'anywhere' }}>
          {t.types.split('>')[0]} … {t.between.replace(/_/g, ' ') || '(adjacent)'} … {t.types.split('>')[1]}{t.right !== '</s>' ? ` ${t.right}` : ''}
        </div>
        <Readouts>
          <Readout label="training candidates" value={String(t.n)} id="tpl-n" />
          <Readout label="ρ at κ = 1 / at this κ" value={`${f3(t.rho)} / ${f3(KAPPA.templates[sel].rho[k])}`} id="tpl-rho" />
          <Readout label={`π (${t.top})`} value={f3(t.pi)} id="tpl-pi" />
          <Readout label="fitted P(relation), κ = 1" value={f3(t.fitted)} id="tpl-fitted" />
          <Readout label="test candidates labeled, this κ" value={f3(KAPPA.templates[sel].labeled[k])} id="tpl-labeled" />
        </Readouts>
      </div>
      <Note>
        Tap a point for its template. At κ = 1 the fitted P(relation) matches ρ within 0.05 on every template (Theorem 1).
        Moving κ left removes related co-mentions from training, which lowers each ρ; the decisions are refits, and π is
        drawn at its κ = 1 value. Removing data raises F1, because that data was teaching the base rate.
      </Note>
    </figure>
  );
}

// ----- C. At least one -----

function WitnessView() {
  const [test, setTest] = useState<Test>('seen');
  const wb = MIL.witness;
  const idf = MIL.identifiability[test];
  const facts = (m: string, agg: 'max' | 'noisy-or') => MODELS[m][test].facts[agg].f1;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>One confident witness silences the rest of the bag</Caption>
      <Sub>
        the bag of ({wb.h}, {wb.t}) for {wb.relation}: MIL&rsquo;s P({wb.relation}), the posterior w that the mention is
        the witness, and Corollary 2&rsquo;s coefficient Π(1 − p_j)/P(∃) on it
      </Sub>
      <div data-panel="bag" style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
        {wb.mentions.map((m, n) => (
          <div key={n} data-mention={n} style={{ borderLeft: `3px solid ${m.stated ? 'var(--color-accent)' : 'var(--color-border)'}`, paddingLeft: '0.5rem' }}>
            <PairLine toks={m.toks} head={m.head} tail={m.tail} id={`bag-${n}`} />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.1rem 0.9rem', fontFamily: mono, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              <span>{m.stated ? 'states it' : 'co-mention'}</span>
              <span>MIL {f3(m.mil)}</span><span>DS {f3(m.ds)}</span><span>w {f3(m.w)}</span>
              <span data-push={n}>coef {m.push.toPrecision(3)}</span>
            </div>
          </div>
        ))}
      </div>
      <Readouts>
        <Readout label="P(∃) under MIL" value={f3(wb.P_any)} id="p-any" />
        <Readout label="at least one fails / DS label error (training)" value={`${f3(MIL.violations.rate)} / ${f3(MIL.violations.ds_error)}`} id="violations" emphasis />
      </Readouts>
      <Sub>what the bag labels determine (Proposition 2), and sentences against facts</Sub>
      <TestPills test={test} setTest={setTest} />
      <Bars id="forced" labelWidth={170} bars={[
        { label: `forced templates (${idf.n_forced})`, v: idf.forced, color: 'var(--color-accent)', id: 'forced' },
        { label: `free templates (${idf.n_free})`, v: idf.free, color: 'var(--color-accent)', id: 'free', faint: true },
      ]} />
      <Note>MIL&rsquo;s recall on stated candidates, by whether some training bag forces their template.</Note>
      <Bars id="levels" labelWidth={170} bars={[
        { label: 'DS, sentence F1', v: MODELS.DS[test].f1, color: MODEL_COLORS.DS, id: 'ds-sent' },
        { label: 'MIL, sentence F1', v: MODELS.MIL[test].f1, color: MODEL_COLORS.MIL, id: 'mil-sent' },
        { label: 'DS, fact F1 (max)', v: facts('DS', 'max'), color: MODEL_COLORS.DS, id: 'ds-max' },
        { label: 'DS, fact F1 (noisy-OR)', v: facts('DS', 'noisy-or'), color: MODEL_COLORS.DS, id: 'ds-nor', faint: true },
        { label: 'MIL, fact F1 (max)', v: facts('MIL', 'max'), color: MODEL_COLORS.MIL, id: 'mil-max', faint: true },
        { label: 'MIL, fact F1 (noisy-OR)', v: facts('MIL', 'noisy-or'), color: MODEL_COLORS.MIL, id: 'mil-nor' },
      ]} />
      <Readouts>
        <Readout label="DS fact precision: max / noisy-OR" value={`${f3(MODELS.DS[test].facts.max.p)} / ${f3(MODELS.DS[test].facts['noisy-or'].p)}`} id="ds-fact-p" />
        <Readout label="MIL sentence P / R" value={`${f3(MODELS.MIL[test].p)} / ${f3(MODELS.MIL[test].r)}`} id="mil-pr" />
      </Readouts>
      <Note>
        Noisy-OR can only exceed the max (Proposition 3). It is MIL&rsquo;s own likelihood, so MIL&rsquo;s probabilities are
        trained for it; for the sentence-trained DS classifier, a pair&rsquo;s repeated co-mentions add up toward 1.
      </Note>
    </figure>
  );
}

// ----- D. Incomplete KB -----

function IncompleteView() {
  const asc = [...KEEP].reverse();
  const key = (c: number) => `${c}`;
  const [k, setK] = useState(asc.length - 3);
  const c = asc[k];
  const r = INC[key(c)];
  const s1: Series[] = [
    { label: 'raw F1', ys: asc.map((v) => INC[key(v)].raw.f1), color: 'var(--color-badge-blue-text)', id: 'raw-f1' },
    { label: 'raw recall', ys: asc.map((v) => INC[key(v)].raw.r), color: 'var(--color-badge-blue-text)', dash: true, id: 'raw-r' },
    { label: 'divided by c: F1', ys: asc.map((v) => INC[key(v)].by_c.f1), color: 'var(--color-accent)', id: 'byc-f1' },
    { label: 'relation mass ratio', ys: asc.map((v) => INC[key(v)].mass_ratio), color: 'var(--color-badge-green-text)', id: 'mass' },
    { label: 'the line c', ys: asc, color: 'var(--color-text-secondary)', dash: true, id: 'c' },
  ];
  const s2: Series[] = [
    { label: 'against the world', ys: asc.map((v) => INC[key(v)].facts_world), color: 'var(--color-accent)', id: 'world' },
    { label: 'against an independent KB of rate c', ys: asc.map((v) => INC[key(v)].facts_indep), color: 'var(--color-badge-blue-text)', id: 'indep' },
    { label: 'against the training KB', ys: asc.map((v) => INC[key(v)].facts_kb), color: 'var(--color-badge-amber-text)', dash: true, id: 'kb' },
    { label: 'c × world precision', ys: asc.map((v) => v * INC[key(v)].facts_world), color: 'var(--color-text-secondary)', dash: true, id: 'c-world' },
  ];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>An incomplete KB scales the optimum by c: it first helps distant supervision, then collapses it</Caption>
      <LineChart xs={asc} series={s1} xLabel="c: share of entity pairs the KB knows (sentence level, seen test)" cursor={c} yDomain={[0, 1]} id="keep-sentences" />
      <Key items={s1.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />
      <Row style={{ marginTop: '0.6rem' }}>
        <Label>c</Label>
        <input type="range" min={0} max={asc.length - 1} step={1} value={k} onChange={(e) => setK(Number(e.target.value))}
          aria-label="keep rate c" data-control="c" style={{ flex: '1 1 10rem', minWidth: 0 }} />
        <span style={{ fontFamily: mono, fontSize: '0.8rem' }} data-readout="c">c = {c.toFixed(1)}</span>
      </Row>
      <Readouts>
        <Readout label="KB facts" value={String(r.kb_facts)} id="kb-facts" />
        <Readout label="relation mass ratio" value={f3(r.mass_ratio)} id="mass" />
        <Readout label="raw P / R / F1" value={`${f3(r.raw.p)} / ${f3(r.raw.r)} / ${f3(r.raw.f1)}`} id="raw" emphasis />
        <Readout label="divided by c: R / F1" value={`${f3(r.by_c.r)} / ${f3(r.by_c.f1)}`} id="by-c" />
        <Readout label="Elkan–Noto e1 / F1 divided by e1" value={`${f3(r.e1)} / ${f3(r.by_e1.f1)}`} id="e1" />
      </Readouts>
      <Sub>fact precision of the incomplete-KB classifier (max aggregation)</Sub>
      <LineChart xs={asc} series={s2} xLabel="c" cursor={c} yDomain={[0, 1]} id="keep-facts" />
      <Key items={s2.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />
      <Readouts>
        <Readout label="world / independent KB / training KB" value={`${f3(r.facts_world)} / ${f3(r.facts_indep)} / ${f3(r.facts_kb)}`} id="fact-p" />
        <Readout label="fact recall (world)" value={f3(r.facts_world_r)} id="fact-r" />
      </Readouts>
      <Note>
        Theorem 3: relation probabilities shrink by c, so a template that always states its relation survives argmax only
        while c &gt; 1/2, and co-mention templates near Corollary 1&rsquo;s threshold drop out first. Proposition 4: scored
        against a KB that also forgets, true facts read as c times their precision. One random KB per c.
      </Note>
    </figure>
  );
}

// ----- E. Pipeline & KB -----

function PipelineView() {
  const [test, setTest] = useState<Test>('unseen');
  const [spans, setSpans] = useState<'gold' | 'predicted'>('predicted');
  const P = PIPE[test][spans];
  const labs = [['gold', 'gold labels'], ['ds', 'DS'], ['mil', 'MIL']] as const;
  const colors = { gold: MODEL_COLORS.gold, ds: MODEL_COLORS.DS, mil: MODEL_COLORS.MIL };
  const miss = PIPE[test].predicted.gold.arg_miss;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>The extractor loses only what the tagger loses: a triple needs both of its arguments</Caption>
      <TestPills test={test} setTest={setTest} />
      <Row>
        <Label>spans</Label>
        <Pill active={spans === 'gold'} onClick={() => setSpans('gold')} data={{ 'data-spans': 'gold' }}>gold spans</Pill>
        <Pill active={spans === 'predicted'} onClick={() => setSpans('predicted')} data={{ 'data-spans': 'predicted' }}>tagger&rsquo;s spans</Pill>
      </Row>
      <Bars id="pipeline" labelWidth={170} marks={[{ x: P.gold.bound, label: 'bound' }]}
        bars={labs.flatMap(([l, name]) => [
          { label: `${name}: triple recall`, v: P[l].r, color: colors[l], id: `${l}-r`, faint: true },
          { label: `${name}: triple F1`, v: P[l].f1, color: colors[l], id: `${l}-f1` },
        ])} />
      <Key items={[{ label: 'Proposition 5’s bound: gold triples with both arguments found', color: 'var(--color-text)', dash: true }]} />
      <Readouts>
        <Readout label="bound" value={f3(P.gold.bound)} id="bound" emphasis />
        {labs.map(([l, name]) => (
          <Readout key={l} label={`${name}: P / R / F1`} value={`${f3(P[l].p)} / ${f3(P[l].r)} / ${f3(P[l].f1)}`} id={`pipe-${l}`} />
        ))}
        <Readout label="MIL facts: P / R" value={`${f3(P.mil.facts.p)} / ${f3(P.mil.facts.r)}`} id="pipe-facts" />
      </Readouts>
      <Sub>argument mentions the tagger missed, by type ({test === 'seen' ? 'seen' : 'held-out'} test)</Sub>
      <div data-panel="arg-miss" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.2rem 1rem', fontFamily: mono, fontSize: '0.78rem' }}>
        {Object.entries(miss).map(([ty, v]) => <span key={ty} data-miss={ty}>{ty} {f3(v)}</span>)}
      </div>
      <Sub>names leak through the arguments: related co-mentions labeled with a relation</Sub>
      <Bars id="memorize" labelWidth={170} bars={(['DS', 'DS + argument words', 'DS + encoder'] as const).flatMap((m) => [
        { label: `${m}, seen`, v: MODELS[m].seen.kinds.related, color: 'var(--color-badge-blue-text)', id: `${m}-seen` },
        { label: `${m}, held out`, v: MODELS[m].unseen.kinds.related, color: 'var(--color-badge-blue-text)', id: `${m}-unseen`, faint: true },
      ])} />
      <Readouts>
        <Readout label="gold labels + encoder: sentence F1 seen / held out" value={`${f3(MODELS['gold + encoder'].seen.f1)} / ${f3(MODELS['gold + encoder'].unseen.f1)}`} id="gold-encoder" />
        <Readout label="test candidates: seen / held out" value={`${SIZES.seen.candidates.toLocaleString('en-US')} / ${SIZES.unseen.candidates.toLocaleString('en-US')}`} id="sizes" />
      </Readouts>
      <Note>
        The relation features never read an argument&rsquo;s words, so on gold spans the held-out companies cost nothing.
        The seen test&rsquo;s bound is lower because its arguments include training-roster names that the tagger&rsquo;s 30
        training documents rarely covered. Argument words, or encoder states at the entity starts, let DS memorize the
        KB pair by pair.
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'labels', label: 'DS labels' },
  { key: 'learns', label: 'What DS learns' },
  { key: 'witness', label: 'At least one' },
  { key: 'incomplete', label: 'Incomplete KB' },
  { key: 'pipeline', label: 'Pipeline & KB' },
] as const;

export default function RelationExtractionLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('learns');
  return (
    <div data-lab="relation-extraction" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'labels' && <LabelsView />}
      {view === 'learns' && <LearnsView />}
      {view === 'witness' && <WitnessView />}
      {view === 'incomplete' && <IncompleteView />}
      {view === 'pipeline' && <PipelineView />}
    </div>
  );
}
