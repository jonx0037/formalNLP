import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/topic-models-lda/topic_models_lda.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_simplex_tab_noise_shrinks_with_length, test_thm1_anchor_rows_are_the_vertices,
// test_anchor_recovery_improves_with_documents, test_the_prior_ranks_the_engines,
// test_finance_anchors_one_per_sector_none_shared, test_mou_oracle_beats_lda_but_em_cannot_find_it,
// test_cavi_stalls_where_its_own_bound_is_lower, test_templates_hide_tone_and_document_type).
// Change the .py, regenerate, never the reverse. Positions the views only draw (a point in a triangle, a line) are
// placed here from payload coordinates; every printed number is a payload value.
import payload from './data/topic-model-lab.json';

/**
 * Topic Model Laboratory — four figures for the topic-models-lda topic.
 *   A. "Topic simplex": three words, three topics; documents' expected and observed distributions, and the KL
 *      projections of the observed ones onto the topic triangle (Proposition 1, Proposition 2).
 *   B. "Anchors": the normalized co-occurrence rows of a three-topic world in its anchors' triangle as the corpus
 *      grows, recovery error against the number of documents, and the finance anchors (Theorem 1, Algorithm 1).
 *   C. "Gibbs vs. VI": the sampler's log joint and coordinate ascent's ELBO, and recovery error per topic, under the
 *      world's own topic prior and a misspecified one (Theorem 2, Theorem 3, Corollary 2).
 *   D. "Finance topics": each method's top words per sector topic, the documents' proportions, and the tables of the
 *      case study.
 */

type Cell = { p: number[]; phat: number[]; proj: number[]; outside: number };
const SIMPLEX = payload.simplex as unknown as { phi: number[]; alphas: number[]; lengths: number[]; cells: Record<string, Cell> };
const ANCH = payload.anchors as unknown as {
  tri: Record<string, number[]>; M: number[]; curve: { sep: number[]; nonsep: number[]; floor: number };
  finance: { a6: [string, number][]; a7: [string, number][] };
};
type GV = { gibbs: number[]; cavi: number[]; l1: Record<Engine, number[]>; mean: Record<Engine, number> };
const ENGINES = ['gibbs', 'cavi', 'anchors'] as const;
type Engine = (typeof ENGINES)[number];
const GVD = payload.gibbs_vi as unknown as Record<'1' | '0.01', GV>;
const METHODS = ['gibbs', 'cavi', 'mou_oracle', 'mou_best', 'anchors'] as const;
type Method = (typeof METHODS)[number];
const FIN = payload.finance as unknown as {
  topics: string[]; words: Record<Method, [string, number][][]>; theta: number[][]; sector: number[];
  table: Record<string, { l1: number; completion?: number; nmi?: number }>;
  templates: Record<'all' | 'band', { sector: number; doc_type: number; tone: number; max_theta: number }>;
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);
const SQ3 = Math.sqrt(3) / 2;

// One color per sector, in fc.SECTORS order; the shared vocabulary is muted.
const SECTOR_COLORS = [
  'var(--color-accent)', 'var(--color-badge-blue-text)', 'var(--color-accent-secondary)',
  'var(--color-badge-amber-text)', 'var(--color-badge-green-text)', 'var(--color-badge-red-text)',
];
const SHARED = 'var(--color-text-secondary)';
const homeColor = (h: number) => (h === 6 ? SHARED : SECTOR_COLORS[h]);

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

function Key({ items }: { items: { label: string; color: string; dash?: boolean; dot?: boolean; hollow?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{
            display: 'inline-block', width: s.dot ? '0.55rem' : '0.9rem', height: s.dash ? 0 : s.dot ? '0.55rem' : '0.2rem',
            borderRadius: s.dot ? '50%' : undefined, boxSizing: 'border-box',
            border: s.hollow ? `1.5px solid ${s.color}` : undefined,
            borderTop: s.dash ? `2px dashed ${s.color}` : s.hollow ? `1.5px solid ${s.color}` : undefined,
            background: s.dash || s.hollow ? undefined : s.color,
          }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** A table that scrolls inside its own box on a narrow screen instead of widening the page. */
function Table({ head, rows, id }: { head: string[]; rows: { cells: ReactNode[]; key: string; strong?: boolean }[]; id: string }) {
  const cell: CSSProperties = { padding: '0.2rem 0.5rem', textAlign: 'right', whiteSpace: 'nowrap', borderBottom: '1px solid var(--color-border)' };
  return (
    <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
      <table data-table={id} style={{ fontFamily: mono, fontSize: '0.75rem', borderCollapse: 'collapse', margin: '0.3rem 0' }}>
        <thead>
          <tr>{head.map((h, i) => <th key={h} style={{ ...cell, fontFamily: sans, fontWeight: 600, textAlign: i === 0 ? 'left' : 'right' }}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-row={r.key}>
              {r.cells.map((c, i) => (
                <td key={i} style={{ ...cell, textAlign: i === 0 ? 'left' : 'right', fontFamily: i === 0 ? sans : mono, fontWeight: r.strong ? 600 : 400 }}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Pairs of a flat [x0, y0, x1, y1, …] list. */
const pairs = (flat: number[]) => d3.range(flat.length / 2).map((i) => [flat[2 * i], flat[2 * i + 1]] as [number, number]);

/** A ternary plot: the triangle with vertices (0, 0), (1, 0), (½, √3/2), with a margin of ``pad`` triangle units on
 *  every side, ``S`` pixels wide; ``H`` is the matching height. */
function ternaryFrame(S: number, pad: number) {
  const H = (S * (SQ3 + 2 * pad)) / (1 + 2 * pad);
  const x = d3.scaleLinear().domain([-pad, 1 + pad]).range([0, S]);
  const y = d3.scaleLinear().domain([-pad, SQ3 + pad]).range([H, 0]);
  return { x, y, H };
}

// ---------------------------------------------------------------------------------------------------------------------
// A. Topic simplex.
// ---------------------------------------------------------------------------------------------------------------------

function SimplexView() {
  const [ai, setAi] = useState(2); // α = 1
  const [ni, setNi] = useState(0); // n_d = 10
  const [show, setShow] = useState<'both' | 'expected' | 'observed'>('both');
  const a = SIMPLEX.alphas[ai];
  const n = SIMPLEX.lengths[ni];
  const cell = SIMPLEX.cells[`${a}/${n}`];
  const S = 340;
  const pad = 0.06;
  const { x, y, H: SH } = ternaryFrame(S, pad);
  const word = [[0, 0], [1, 0], [0.5, SQ3]] as [number, number][];
  const topics = pairs(SIMPLEX.phi);
  const P = pairs(cell.p);
  const H = pairs(cell.phat);
  const J = pairs(cell.proj);
  const poly = (pts: [number, number][]) => pts.map(([u, v]) => `${x(u)},${y(v)}`).join(' ');
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Documents live in the topic triangle only in expectation</Caption>
      <Row>
        <Label>prior α</Label>
        {SIMPLEX.alphas.map((v, i) => <Pill key={v} active={ai === i} onClick={() => setAi(i)} data={{ 'data-alpha': `${v}` }}>{v}</Pill>)}
      </Row>
      <Row>
        <Label>words per document n_d</Label>
        {SIMPLEX.lengths.map((v, i) => <Pill key={v} active={ni === i} onClick={() => setNi(i)} data={{ 'data-length': `${v}` }}>{v}</Pill>)}
      </Row>
      <Row>
        <Label>show</Label>
        {(['both', 'expected', 'observed'] as const).map((v) => <Pill key={v} active={show === v} onClick={() => setShow(v)} data={{ 'data-show': v }}>{v}</Pill>)}
      </Row>
      <svg viewBox={`0 0 ${S} ${SH}`} style={{ width: '100%', maxWidth: '26rem', height: 'auto', display: 'block' }} role="img" data-panel="simplex"
        aria-label="The word simplex for three words, the topic triangle inside it, and documents as points">
        <polygon points={poly(word)} fill="none" stroke="var(--color-border)" strokeWidth={1.5} />
        <polygon points={poly(topics)} fill="var(--color-accent)" fillOpacity={0.08} stroke="var(--color-accent)" strokeWidth={1.5} />
        {['word 1', 'word 2', 'word 3'].map((t, i) => (
          <text key={t} x={x(word[i][0]) + (i === 0 ? 2 : i === 1 ? -2 : 0)} y={y(word[i][1]) + (i === 2 ? -6 : 14)}
            textAnchor={i === 0 ? 'start' : i === 1 ? 'end' : 'middle'} style={axisText}>{t}</text>
        ))}
        {show !== 'expected' && H.map(([u, v], i) => (
          <line key={`l${i}`} x1={x(u)} y1={y(v)} x2={x(J[i][0])} y2={y(J[i][1])} stroke="var(--color-text-secondary)" strokeWidth={0.6} strokeOpacity={0.6} />
        ))}
        {show !== 'observed' && P.map(([u, v], i) => <circle key={`p${i}`} cx={x(u)} cy={y(v)} r={2.4} fill="var(--color-accent-secondary)" />)}
        {show !== 'expected' && H.map(([u, v], i) => (
          <circle key={`h${i}`} cx={x(u)} cy={y(v)} r={2.6} fill="none" stroke="var(--color-accent)" strokeWidth={1} />
        ))}
        {topics.map(([u, v], i) => <circle key={`t${i}`} cx={x(u)} cy={y(v)} r={4} fill="var(--color-accent)" />)}
      </svg>
      <Key items={[
        { label: 'topics φ_k and their triangle', color: 'var(--color-accent)', dot: true },
        { label: 'expected p_d = Φᵀθ_d', color: 'var(--color-accent-secondary)', dot: true },
        { label: 'observed frequencies p̂_d', color: 'var(--color-accent)', dot: true, hollow: true },
        { label: 'KL projection onto the triangle', color: 'var(--color-text-secondary)' },
      ]} />
      <Readouts>
        <Readout label="observed documents outside the topic triangle" value={f3(cell.outside)} id="outside" emphasis />
      </Readouts>
      <Note>80 documents per setting; the same θ_d at every length, so only the sampling noise changes. Small α piles documents at the topics; short documents scatter their observed frequencies outside, and Proposition 2 maps each back to the nearest point in KL divergence.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// B. Anchors.
// ---------------------------------------------------------------------------------------------------------------------

function AnchorsView() {
  const keys = [...ANCH.M.map((m) => `${m}`), 'population'];
  const [ki, setKi] = useState(0);
  const key = keys[ki];
  const S = 340;
  const pad = 0.25;
  const { x, y, H: SH } = ternaryFrame(S, pad);
  const tri = [[0, 0], [1, 0], [0.5, SQ3]] as [number, number][];
  const pts = pairs(ANCH.tri[key]);
  const poly = tri.map(([u, v]) => `${x(u)},${y(v)}`).join(' ');
  // recovery-vs-M chart
  const CW = 320, CH = 200, m = { l: 40, r: 10, t: 10, b: 30 };
  const cx = d3.scaleLog().domain([200, 20000]).range([m.l, CW - m.r]);
  const cy = d3.scaleLinear().domain([0, 0.6]).range([CH - m.b, m.t]);
  const line = (vals: number[]) => d3.line<number>().x((_, i) => cx(ANCH.M[i])).y((v) => cy(v))(vals) ?? '';
  const mi = ki < ANCH.M.length ? ki : -1;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Co-occurrence rows in the anchors' triangle (three-topic world; clipped at the frame)</Caption>
      <Row>
        <Label>documents M</Label>
        {keys.map((k, i) => <Pill key={k} active={ki === i} onClick={() => setKi(i)} data={{ 'data-m': k }}>{k === 'population' ? 'population' : Number(k).toLocaleString('en-US')}</Pill>)}
      </Row>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
        <svg viewBox={`0 0 ${S} ${SH}`} style={{ flex: '1 1 16rem', maxWidth: '22rem', height: 'auto', display: 'block' }} role="img" data-panel="anchors-tri"
          aria-label="Each word's normalized co-occurrence row, in coordinates on the three anchor rows">
          <defs><clipPath id="tm-anchor-clip"><rect x={0} y={0} width={S} height={SH} /></clipPath></defs>
          <polygon points={poly} fill="var(--color-accent)" fillOpacity={0.06} stroke="var(--color-accent)" strokeWidth={1.5} />
          <g clipPath="url(#tm-anchor-clip)">
            {pts.map(([u, v], i) => (
              <circle key={i} cx={x(u)} cy={y(v)} r={i < 3 ? 4.5 : 2.4}
                fill={i < 3 ? 'var(--color-accent)' : 'var(--color-accent-secondary)'} fillOpacity={i < 3 ? 1 : 0.8} />
            ))}
          </g>
          {['anchor 1', 'anchor 2', 'anchor 3'].map((t, i) => (
            <text key={t} x={x(tri[i][0])} y={y(tri[i][1]) + (i === 2 ? -8 : 16)} textAnchor="middle" style={axisText}>{t}</text>
          ))}
        </svg>
        <svg viewBox={`0 0 ${CW} ${CH}`} style={{ flex: '1 1 16rem', maxWidth: '22rem', height: 'auto', display: 'block' }} role="img" data-panel="anchors-curve"
          aria-label="Mean recovery error per topic against the number of documents, with and without anchor words">
          {[0, 0.2, 0.4, 0.6].map((v) => (
            <g key={v}>
              <line x1={m.l} x2={CW - m.r} y1={cy(v)} y2={cy(v)} stroke="var(--color-border)" strokeWidth={0.5} />
              <text x={m.l - 4} y={cy(v) + 3} textAnchor="end" style={axisText}>{v.toFixed(1)}</text>
            </g>
          ))}
          {ANCH.M.map((v) => <text key={v} x={cx(v)} y={CH - m.b + 14} textAnchor="middle" style={axisText}>{v >= 1000 ? `${v / 1000}k` : v}</text>)}
          <text x={(m.l + CW - m.r) / 2} y={CH - 2} textAnchor="middle" style={axisText}>documents (log scale)</text>
          <line x1={m.l} x2={CW - m.r} y1={cy(ANCH.curve.floor)} y2={cy(ANCH.curve.floor)} stroke="var(--color-badge-amber-text)" strokeDasharray="4 3" />
          <path d={line(ANCH.curve.nonsep)} fill="none" stroke="var(--color-badge-amber-text)" strokeWidth={2} />
          <path d={line(ANCH.curve.sep)} fill="none" stroke="var(--color-accent)" strokeWidth={2} />
          {mi >= 0 && <line x1={cx(ANCH.M[mi])} x2={cx(ANCH.M[mi])} y1={m.t} y2={CH - m.b} stroke="var(--color-text-secondary)" strokeWidth={0.6} />}
        </svg>
      </div>
      <Key items={[
        { label: 'anchor rows', color: 'var(--color-accent)', dot: true },
        { label: 'other words', color: 'var(--color-accent-secondary)', dot: true },
        { label: 'separable world (V = 400, K = 5)', color: 'var(--color-accent)' },
        { label: 'no anchors', color: 'var(--color-badge-amber-text)' },
        { label: 'no-anchor floor at infinite data', color: 'var(--color-badge-amber-text)', dash: true },
      ]} />
      <Readouts>
        {mi >= 0 && <Readout label="mean L¹ error, separable" value={f3(ANCH.curve.sep[mi])} id="sep" emphasis />}
        {mi >= 0 && <Readout label="mean L¹ error, no anchors" value={f3(ANCH.curve.nonsep[mi])} id="nonsep" />}
        <Readout label="no-anchor floor" value={f3(ANCH.curve.floor)} id="floor" />
      </Readouts>
      <Sub>Finance (restaurant vocabulary): FastAnchorWords' picks and the vocabulary each word was drawn from</Sub>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.8rem', fontFamily: sans, fontSize: '0.8rem' }} data-anchors="finance">
        {ANCH.finance.a7.map(([w, h], i) => (
          <span key={w} style={{ color: homeColor(h), fontWeight: i === 6 ? 400 : 600 }}>
            {i === 6 ? '7th: ' : ''}{w} <span style={{ color: 'var(--color-text-secondary)', fontWeight: 400 }}>({FIN.topics[h]})</span>
          </span>
        ))}
      </div>
      <Note>With population co-occurrences the rows fill the triangle exactly (Theorem 1). Sampled rows scatter outside and tighten as the corpus grows. In the finance corpus six anchors land one per sector; the seventh is another industrials word, never the shared vocabulary, which every document carries at weight ½ (Proposition 4).</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// C. Gibbs vs. VI.
// ---------------------------------------------------------------------------------------------------------------------

function Trace({ values, label, id, xLabel, color }: { values: number[]; label: string; id: string; xLabel: string; color: string }) {
  const CW = 320, CH = 190, m = { l: 62, r: 10, t: 10, b: 30 };
  const [lo, hi] = d3.extent(values) as [number, number];
  const cx = d3.scaleLinear().domain([1, values.length]).range([m.l, CW - m.r]);
  const cy = d3.scaleLinear().domain([lo, hi]).nice(4).range([CH - m.b, m.t]);
  const path = d3.line<number>().x((_, i) => cx(i + 1)).y((v) => cy(v))(values) ?? '';
  return (
    <svg viewBox={`0 0 ${CW} ${CH}`} style={{ flex: '1 1 16rem', maxWidth: '22rem', height: 'auto', display: 'block' }} role="img" data-panel={id} aria-label={label}>
      {cy.ticks(4).map((v) => (
        <g key={v}>
          <line x1={m.l} x2={CW - m.r} y1={cy(v)} y2={cy(v)} stroke="var(--color-border)" strokeWidth={0.5} />
          <text x={m.l - 4} y={cy(v) + 3} textAnchor="end" style={axisText}>{d3.format(',')(v)}</text>
        </g>
      ))}
      {cx.ticks(4).map((v) => <text key={v} x={cx(v)} y={CH - m.b + 14} textAnchor="middle" style={axisText}>{v}</text>)}
      <text x={(m.l + CW - m.r) / 2} y={CH - 2} textAnchor="middle" style={axisText}>{xLabel}</text>
      <text x={m.l + 4} y={m.t + 10} style={{ ...axisText, fill: 'var(--color-text)' }}>{label}</text>
      <path d={path} fill="none" stroke={color} strokeWidth={1.8} />
    </svg>
  );
}

function GibbsViView() {
  const [beta, setBeta] = useState<'1' | '0.01'>('1');
  const g = GVD[beta];
  const names: Record<Engine, string> = { gibbs: 'collapsed Gibbs', cavi: 'coordinate ascent', anchors: 'anchors' };
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Two engines and the anchors on one separable world (V = 400, K = 5, 500 documents)</Caption>
      <Row>
        <Label>topic prior β</Label>
        <Pill active={beta === '1'} onClick={() => setBeta('1')} data={{ 'data-beta': '1' }}>1 (the world's own)</Pill>
        <Pill active={beta === '0.01'} onClick={() => setBeta('0.01')} data={{ 'data-beta': '0.01' }}>0.01 (misspecified)</Pill>
      </Row>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem' }}>
        <Trace values={g.gibbs} label="log p(w, z), per sweep" id="gibbs-trace" xLabel="sweep" color="var(--color-accent)" />
        <Trace values={g.cavi} label="ELBO, per iteration" id="cavi-trace" xLabel="iteration" color="var(--color-badge-blue-text)" />
      </div>
      <Sub>L¹ error of each recovered topic, after matching topics by least total L¹</Sub>
      <Table id="gv-l1" head={['topic', ...ENGINES.map((e) => names[e])]}
        rows={[
          ...g.l1.gibbs.map((_, k) => ({ key: `t${k}`, cells: [`${k + 1}`, ...ENGINES.map((e) => f3(g.l1[e][k]))] })),
          { key: 'mean', strong: true, cells: ['mean', ...ENGINES.map((e) => f3(g.mean[e]))] },
        ]} />
      <Note>The two traces are different quantities: the sampler's log joint wanders at its posterior's level, while the ELBO only rises (Corollary 2). The anchors do not depend on β. At the world's own prior the sampler recovers the topics best; at a sparse, misspecified one the order reverses, which is Asuncion et al.'s point that smoothing explains most differences between engines.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// D. Finance topics.
// ---------------------------------------------------------------------------------------------------------------------

const METHOD_NAMES: Record<string, string> = {
  gibbs: 'collapsed Gibbs', cavi: 'coordinate ascent from noise', cavi_warm: "coordinate ascent from Gibbs's counts",
  anchors: 'anchors (Algorithm 1)', mou_best: 'mixture of unigrams, best of 20 starts', mou_oracle: 'mixture of unigrams from the sectors',
};

function ThetaStrip() {
  const n = FIN.theta.length;
  const CW = 640, CH = 90, top = 16;
  const bw = CW / n;
  const groups = d3.range(6).map((s) => {
    const idx = FIN.sector.map((v, i) => (v === s ? i : -1)).filter((i) => i >= 0);
    return { s, from: idx[0], to: idx[idx.length - 1] + 1 };
  });
  return (
    <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', height: 'auto', display: 'block' }} role="img" data-panel="theta"
      aria-label="Each training document's topic proportions under collapsed Gibbs, documents grouped by sector">
      {FIN.theta.map((row, i) => {
        let acc = 0;
        return (
          <g key={i}>
            {row.map((v, k) => {
              const h = v * (CH - top);
              const el = <rect key={k} x={i * bw} y={top + acc} width={bw + 0.3} height={h} fill={SECTOR_COLORS[k]} />;
              acc += h;
              return el;
            })}
          </g>
        );
      })}
      {groups.map((g) => (
        <g key={g.s}>
          <line x1={g.from * bw} x2={g.from * bw} y1={0} y2={CH} stroke="var(--color-bg)" strokeWidth={1.5} />
          {g.to - g.from >= 25 && (
            <text x={((g.from + g.to) / 2) * bw} y={11} textAnchor="middle" style={axisText}>{FIN.topics[g.s]}</text>
          )}
        </g>
      ))}
    </svg>
  );
}

function FinanceView() {
  const [method, setMethod] = useState<Method>('gibbs');
  const words = FIN.words[method];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>The six identifiable finance topics, as each method recovers them</Caption>
      <Row>
        <Label>method</Label>
        {METHODS.map((m) => <Pill key={m} active={method === m} onClick={() => setMethod(m)} data={{ 'data-method': m }}>{METHOD_NAMES[m]}</Pill>)}
      </Row>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }} data-words={method}>
        {words.map((ws, k) => (
          <div key={k} style={{ flex: '1 1 8.5rem', minWidth: 0, border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.4rem 0.5rem' }}>
            <div style={{ fontFamily: sans, fontSize: '0.75rem', fontWeight: 600, color: SECTOR_COLORS[k], marginBottom: '0.2rem' }}>{FIN.topics[k]}</div>
            {ws.map(([w, h]) => (
              <div key={w} style={{ fontFamily: mono, fontSize: '0.72rem', color: homeColor(h), fontStyle: h === 6 ? 'italic' : 'normal', overflowWrap: 'anywhere' }}>{w}</div>
            ))}
          </div>
        ))}
      </div>
      <Key items={[
        ...FIN.topics.slice(0, 6).map((t, k) => ({ label: t, color: SECTOR_COLORS[k] })),
        { label: 'shared vocabulary (italic)', color: SHARED },
      ]} />
      <Sub>Each training document's proportions under collapsed Gibbs, documents grouped by their sector</Sub>
      <ThetaStrip />
      <Sub>The engines on the six identifiable topics (three seeds; completion in nats per held-out token)</Sub>
      <Table id="engines" head={['method', 'L¹', 'completion', 'sector NMI']}
        rows={['gibbs', 'cavi', 'cavi_warm', 'anchors', 'mou_best', 'mou_oracle'].map((k) => {
          const r = FIN.table[k];
          return {
            key: k, strong: k === method,
            cells: [METHOD_NAMES[k], f3(r.l1), r.completion === undefined ? '—' : f3(r.completion), r.nmi === undefined ? '—' : f3(r.nmi)],
          };
        })} />
      <Sub>With the template text kept: NMI of each document's dominant topic with its labels</Sub>
      <Table id="templates" head={['vocabulary', 'sector', 'document type', 'tone']}
        rows={(['all', 'band'] as const).map((k) => ({
          key: k, cells: [k === 'all' ? 'every word' : 'words in 3–90 documents', f3(FIN.templates[k].sector), f3(FIN.templates[k].doc_type), f3(FIN.templates[k].tone)],
        }))} />
      <Note>Word colors give the vocabulary each word was drawn from. Each identifiable topic is half shared vocabulary (Proposition 4), so shared words, in italics, fill much of every list; a word in another sector's color marks mixing. The runs started from noise, coordinate ascent and the best EM start, mix sectors most, and even the Gibbs run shown here (seed 0) has no energy word among its energy topic's top ten.</Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------

const VIEWS = [
  { key: 'simplex', label: 'Topic simplex' },
  { key: 'anchors', label: 'Anchors' },
  { key: 'engines', label: 'Gibbs vs. VI' },
  { key: 'finance', label: 'Finance topics' },
] as const;

export default function TopicModelLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('simplex');
  return (
    <div data-lab="topic-model" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'simplex' && <SimplexView />}
      {view === 'anchors' && <AnchorsView />}
      {view === 'engines' && <GibbsViView />}
      {view === 'finance' && <FinanceView />}
    </div>
  );
}
