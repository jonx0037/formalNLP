import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/knowledge-graph-embeddings/knowledge_graph_embeddings.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_thm1_rotation_patterns_are_phase_equations,
// test_thm1_trained_phases_obey_the_equations_and_incompleteness_breaks_them,
// test_thm2_isometries_collapse_n_to_1_heads_symmetric_pairs_and_odd_cycles,
// test_thm3_abelian_models_cannot_separate_s3s_corruptions, test_cor1_the_dihedral_group_fits_s3_exactly,
// test_pattern_results_the_page_reports, test_completing_the_extracted_kb).
// Change the .py, regenerate, never the reverse. Positions the views only draw (a rotated point, a sum of phases) are
// computed here from payload parameters; every printed number is a payload value.
import payload from './data/knowledge-graph-embedding-lab.json';

/**
 * Knowledge-Graph Embedding Laboratory — five figures for the knowledge-graph-embeddings topic.
 *   A. "Translate or rotate": a symmetric pair and a symmetric triangle in the plane, moved by a rotation or a
 *      translation (Proposition 1(c), Theorem 1(a), Theorem 2(c)).
 *   B. "Relation phases": the trained RotatE's phases on the unit circle, on the full KB and with facts hidden
 *      (Theorem 1, and the case study's closed-world finding).
 *   C. "Non-commuting": the Cayley graph of S₃ under D₃ and under the trained abelian models (Theorem 3, Corollary 1).
 *   D. "Link prediction": filtered ranks for every hidden fact, per model, and the pattern table.
 *   E. "Completing extraction": Topic 23's extracted facts about the held-out companies and the missing facts' ranks.
 */

type MS = [number, number];
const ENTS = payload.entities as string[];
const MODELS = ['TransE', 'RotatE', 'DistMult', 'ComplEx'] as const;
type Model = (typeof MODELS)[number];
const PATTERNS = ['symmetry', 'inverse', 'composition', 'inheritance'] as const;
type Pattern = (typeof PATTERNS)[number];

type ToyCase = { points: [number, number][]; facts: [number, number][]; rotation: number[][]; translation: number[][] };
const TOY = payload.toy as unknown as { pair: ToyCase; triangle: ToyCase; theta: number[]; shift: number[] };
type PhaseSet = { theta: Record<string, number[]>; weight: number[]; residual: Record<string, number[]>; median: Record<string, number> };
const PHASES = payload.phases as unknown as { full: PhaseSet; split: PhaseSet; uniform_median: Record<string, number> };
type Abelian = { eps: number; gamma_min: number; ratio: number; points?: number[][] };
const S3 = payload.s3 as unknown as {
  entities: string[]; facts: [number, string, number][];
  dihedral: { points: [number, number][]; eps: number; separation: number; gamma_min: number };
  abelian: Record<'TransE' | 'RotatE', Record<string, Abelian>>;
};
type PatternRow = Record<Pattern, Record<'mrr' | 'h1' | 'h3' | 'h10', MS>> & { direction: MS; train_mrr: MS };
const PAT = payload.patterns as unknown as Record<Model, PatternRow>;
const CONFIGS = payload.configs as unknown as Record<Model, { gamma: number; lam: number; steps: number }>;
type Query = { pattern: Pattern; h: number; r: string; t: number; models: Record<Model, { top: number[]; rank: number }> };
const QUERIES = (payload.queries as unknown as { rows: Query[] }).rows;
type Fact = { h: number; r: string; t: number; score: number };
type Completion = {
  extracted: (Fact & { true: boolean })[]; missing: (Fact & { ranks: Record<Model, number> })[]; known: number; true: number;
  metrics: Record<Model, Record<string, { mrr: MS; h10: MS }>>;
};
const COMP = payload.completion as unknown as Record<'0.5' | '0.1', Completion>;
const EX = payload.example as Fact;

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);
const ms = (v: MS) => `${f3(v[0])} ± ${f3(v[1])}`;
const W = 640;

const MODEL_COLORS: Record<Model, string> = {
  TransE: 'var(--color-badge-amber-text)', RotatE: 'var(--color-accent)', DistMult: 'var(--color-badge-blue-text)',
  ComplEx: 'var(--color-badge-green-text)',
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

function ModelPills({ model, setModel }: { model: Model; setModel: (m: Model) => void }) {
  return (
    <Row>
      <Label>model</Label>
      {MODELS.map((m) => <Pill key={m} active={model === m} onClick={() => setModel(m)} data={{ 'data-model': m }}>{m}</Pill>)}
    </Row>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// A. Translate or rotate.
// ---------------------------------------------------------------------------------------------------------------------

const ENT_NAMES = ['a', 'b', 'c'];

function ToyView() {
  const [shape, setShape] = useState<'pair' | 'triangle'>('pair');
  const [mode, setMode] = useState<'rotate' | 'translate'>('rotate');
  const [ti, setTi] = useState(18); // 90°
  const [si, setSi] = useState(30); // s = 1.0
  const C = TOY[shape];
  const P = C.points;
  const idx = mode === 'rotate' ? ti : si;
  const res = (mode === 'rotate' ? C.rotation : C.translation)[idx];
  const worst = Math.max(...res);
  const th = (TOY.theta[ti] * Math.PI) / 180;
  const axis = [P[1][0] - P[0][0], P[1][1] - P[0][1]];
  const len = Math.hypot(axis[0], axis[1]);
  const move = (p: [number, number]): [number, number] => mode === 'rotate'
    ? [p[0] * Math.cos(th) - p[1] * Math.sin(th), p[0] * Math.sin(th) + p[1] * Math.cos(th)]
    : [p[0] + (TOY.shift[si] * axis[0]) / len, p[1] + (TOY.shift[si] * axis[1]) / len];
  const S = 300, R = 2.4;
  const sc = d3.scaleLinear().domain([-R, R]).range([0, S]);
  const sy = d3.scaleLinear().domain([-R, R]).range([S, 0]);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>A relation as a map of the plane: does it carry each head onto its tail?</Caption>
      <Row>
        <Label>relation holds both ways on</Label>
        <Pill active={shape === 'pair'} onClick={() => setShape('pair')} data={{ 'data-shape': 'pair' }}>a pair</Pill>
        <Pill active={shape === 'triangle'} onClick={() => setShape('triangle')} data={{ 'data-shape': 'triangle' }}>a triangle</Pill>
      </Row>
      <Row>
        <Label>map</Label>
        <Pill active={mode === 'rotate'} onClick={() => setMode('rotate')} data={{ 'data-mode': 'rotate' }}>rotate (RotatE)</Pill>
        <Pill active={mode === 'translate'} onClick={() => setMode('translate')} data={{ 'data-mode': 'translate' }}>translate (TransE)</Pill>
      </Row>
      <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: '22rem', height: 'auto', display: 'block' }} role="img" data-panel="toy"
        aria-label={`largest residual ${f3(worst)}`}>
        <line x1={sc(-R)} x2={sc(R)} y1={sy(0)} y2={sy(0)} style={{ stroke: 'var(--color-border)' }} />
        <line x1={sc(0)} x2={sc(0)} y1={sy(-R)} y2={sy(R)} style={{ stroke: 'var(--color-border)' }} />
        <circle cx={sc(0)} cy={sy(0)} r={sc(1) - sc(0)} style={{ fill: 'none', stroke: 'var(--color-border)', strokeDasharray: '3 3' }} />
        {C.facts.map(([i, j], k) => {
          const m = move(P[i]);
          return (
            <g key={k}>
              <line x1={sc(m[0])} y1={sy(m[1])} x2={sc(P[j][0])} y2={sy(P[j][1])}
                style={{ stroke: 'var(--color-badge-blue-text)', strokeWidth: 1.2, strokeDasharray: '4 3' }} />
              <circle cx={sc(m[0])} cy={sy(m[1])} r={4} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 1.5 }} />
            </g>
          );
        })}
        {P.map((p, i) => (
          <g key={i}>
            <circle cx={sc(p[0])} cy={sy(p[1])} r={5} style={{ fill: 'var(--color-text)' }} />
            <text x={sc(p[0]) + 8} y={sy(p[1]) - 6} style={{ ...axisText, fontSize: '12px', fill: 'var(--color-text)' }}>{ENT_NAMES[i]}</text>
          </g>
        ))}
      </svg>
      <Key items={[
        { label: 'entity', color: 'var(--color-text)', dot: true },
        { label: 'head moved by the relation', color: 'var(--color-accent)', dot: true },
        { label: 'residual to the tail', color: 'var(--color-badge-blue-text)', dash: true },
      ]} />
      <Row style={{ marginTop: '0.6rem' }}>
        <Label>{mode === 'rotate' ? 'θ' : 's'}</Label>
        {mode === 'rotate' ? (
          <input type="range" min={0} max={TOY.theta.length - 1} step={1} value={ti} onChange={(e) => setTi(Number(e.target.value))}
            aria-label="rotation angle" data-control="theta" style={{ flex: '1 1 10rem', minWidth: 0 }} />
        ) : (
          <input type="range" min={0} max={TOY.shift.length - 1} step={1} value={si} onChange={(e) => setSi(Number(e.target.value))}
            aria-label="translation along a to b" data-control="shift" style={{ flex: '1 1 10rem', minWidth: 0 }} />
        )}
        <span style={{ fontFamily: mono, fontSize: '0.8rem' }} data-readout="param">
          {mode === 'rotate' ? `θ = ${TOY.theta[ti]}°` : `s = ${TOY.shift[si].toFixed(1)}`}
        </span>
      </Row>
      <Readouts>
        <Readout label="largest residual" value={f3(worst)} id="worst" emphasis />
        {C.facts.map(([i, j], k) => (
          <Readout key={k} label={`${ENT_NAMES[i]} → ${ENT_NAMES[j]}`} value={f3(res[k])} id={`res-${ENT_NAMES[i]}${ENT_NAMES[j]}`} />
        ))}
      </Readouts>
      <Note>
        {shape === 'pair'
          ? 'With b = −a, rotating by 180° carries a to b and b back to a (Theorem 1(a)). A translation moves both the same way, so the two residuals add to twice the shift and subtract to twice b − a: the larger is never below |b − a| = 2 (Proposition 1(c)).'
          : 'Each pair of the triangle is related both ways. Rotating by 120° fixes a → b but breaks b → a; no angle zeroes all six residuals, because an odd cycle of a symmetric relation collapses under any isometry (Theorem 2(c)).'}
      </Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// B. Relation phases.
// ---------------------------------------------------------------------------------------------------------------------

const wrap = (x: number) => ((((x + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;

function PhaseCircle({ title, series, weight, id, marks }: {
  title: string; series: { phases: number[]; color: string; hollow?: boolean }[]; weight: number[]; id: string; marks?: number[];
}) {
  const S = 150, r0 = 55, c = S / 2;
  const wmax = Math.max(...weight);
  return (
    <div style={{ flex: '1 1 9rem', minWidth: 0, maxWidth: '12rem' }}>
      <div style={{ fontFamily: sans, fontSize: '0.72rem', textAlign: 'center' }}>{title}</div>
      <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id} aria-label={title}>
        <circle cx={c} cy={c} r={r0} style={{ fill: 'none', stroke: 'var(--color-border)' }} />
        <line x1={c - r0 - 6} x2={c + r0 + 6} y1={c} y2={c} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
        {(marks ?? []).map((m) => (
          <circle key={m} cx={c + r0 * Math.cos(m)} cy={c - r0 * Math.sin(m)} r={7} style={{ fill: 'none', stroke: 'var(--color-text-secondary)', strokeDasharray: '2 2' }} />
        ))}
        {series.map((s, si) => s.phases.map((p, i) => (
          <circle key={`${si}-${i}`} cx={c + r0 * Math.cos(p)} cy={c - r0 * Math.sin(p)} r={2 + 3 * (weight[i] / wmax)}
            style={{ fill: s.hollow ? 'none' : s.color, stroke: s.color, strokeWidth: 1.2, fillOpacity: 0.7 }} />
        )))}
        <text x={c + r0 + 8} y={c + 3} style={axisText}>0</text>
        <text x={c - r0 - 16} y={c + 3} style={axisText}>π</text>
      </svg>
    </div>
  );
}

function PhasesView() {
  const [which, setWhich] = useState<'full' | 'split'>('full');
  const P = PHASES[which];
  const T = P.theta;
  const neg = (xs: number[]) => xs.map((x) => -x);
  const sum = T.HQ_IN.map((x, i) => wrap(x + T.LOCATED_IN[i]));
  const keys = ['symmetry', 'inverse', 'composition', 'implication'] as const;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>RotatE&rsquo;s sixteen learned phases per relation, sized by how much of the entities lives on each coordinate</Caption>
      <Row>
        <Label>trained on</Label>
        <Pill active={which === 'full'} onClick={() => setWhich('full')} data={{ 'data-kb': 'full' }}>the full KB</Pill>
        <Pill active={which === 'split'} onClick={() => setWhich('split')} data={{ 'data-kb': 'split' }}>the KB with facts hidden</Pill>
      </Row>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center' }}>
        <PhaseCircle id="ph-sym" title="COMPETES_WITH: at 0 or π?" weight={P.weight} marks={[0, Math.PI]}
          series={[{ phases: T.COMPETES_WITH, color: 'var(--color-accent)' }]} />
        <PhaseCircle id="ph-inv" title="PARENT_OF vs −SUBSIDIARY_OF" weight={P.weight}
          series={[{ phases: T.PARENT_OF, color: 'var(--color-accent)' }, { phases: neg(T.SUBSIDIARY_OF), color: 'var(--color-badge-blue-text)', hollow: true }]} />
        <PhaseCircle id="ph-comp" title="BASED_IN vs HQ_IN + LOCATED_IN" weight={P.weight}
          series={[{ phases: T.BASED_IN, color: 'var(--color-accent)' }, { phases: sum, color: 'var(--color-badge-blue-text)', hollow: true }]} />
        <PhaseCircle id="ph-impl" title="ACQUIRED vs PARENT_OF" weight={P.weight}
          series={[{ phases: T.ACQUIRED, color: 'var(--color-accent)' }, { phases: T.PARENT_OF, color: 'var(--color-badge-blue-text)', hollow: true }]} />
      </div>
      <Key items={[
        { label: 'first relation', color: 'var(--color-accent)', dot: true },
        { label: 'what Theorem 1 says it should equal', color: 'var(--color-badge-blue-text)', dot: true },
      ]} />
      <Sub>median residual of Theorem 1&rsquo;s equation (random phases in brackets)</Sub>
      <Readouts>
        {keys.map((k) => (
          <Readout key={k} label={`${k} (${f3(PHASES.uniform_median[k])})`} value={f3(P.median[k])} id={`med-${k}`} emphasis={k === 'symmetry'} />
        ))}
      </Readouts>
      <Note>
        {which === 'full'
          ? 'On the full knowledge base the phases solve all three equations: COMPETES_WITH sits at 0 or π, PARENT_OF is the negative of SUBSIDIARY_OF, and BASED_IN is the sum of HQ_IN and LOCATED_IN.'
          : 'With one direction of every competitor pair hidden, each hidden direction is a training negative. A rotation with phases at 0 or π scores it exactly like its twin, so training moves the phases off: RotatE learns the antisymmetry the incomplete knowledge base teaches.'}
      </Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// C. Non-commuting.
// ---------------------------------------------------------------------------------------------------------------------

function CayleyPanel({ title, points, id }: { title: string; points: number[][]; id: string }) {
  const S = 170, pad = 22;
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const span = Math.max(d3.max(xs)! - d3.min(xs)!, d3.max(ys)! - d3.min(ys)!, 1e-6);
  const cx = (d3.max(xs)! + d3.min(xs)!) / 2, cy = (d3.max(ys)! + d3.min(ys)!) / 2;
  const k = (S - 2 * pad) / span;
  const collapsed = span < 1e-3;
  const X = (p: number[]) => S / 2 + (p[0] - cx) * k, Y = (p: number[]) => S / 2 - (p[1] - cy) * k;
  return (
    <div style={{ flex: '1 1 9rem', minWidth: 0, maxWidth: '13rem' }}>
      <div style={{ fontFamily: sans, fontSize: '0.72rem', textAlign: 'center' }}>{title}</div>
      {collapsed && (
        <div data-collapsed={id} style={{ fontFamily: sans, fontSize: '0.7rem', textAlign: 'center', color: 'var(--color-text-secondary)' }}>
          all six entities at one point
        </div>
      )}
      <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id} aria-label={title}>
        {!collapsed && S3.facts.map(([h, r, t], i) => (
          <line key={i} x1={X(points[h])} y1={Y(points[h])} x2={X(points[t])} y2={Y(points[t])}
            style={{ stroke: r === 'A' ? 'var(--color-accent)' : 'var(--color-badge-blue-text)', strokeWidth: 1.2, strokeDasharray: r === 'A' ? undefined : '4 3', opacity: 0.8 }} />
        ))}
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={X(p)} cy={Y(p)} r={4} style={{ fill: 'var(--color-text)' }} />
            {!collapsed && <text x={X(p) + 5} y={Y(p) - 5} style={{ ...axisText, fontSize: '9px' }}>{S3.entities[i]}</text>}
          </g>
        ))}
      </svg>
    </div>
  );
}

function S3View() {
  const D = S3.dihedral;
  const runs = (['TransE', 'RotatE'] as const).flatMap((m) => ['1', '2', '8'].map((k) => ({ m, k, a: S3.abelian[m][k] })));
  const rows: { key: string; cells: string[]; strong?: boolean }[] = runs.map(({ m, k, a }) => (
    { key: `${m}-${k}`, cells: [`${m}, k = ${k}`, f3(a.eps), f3(a.gamma_min), f3(a.ratio)] }));
  rows.push({ key: 'D3', cells: ['D₃ (Corollary 1)', f3(D.eps), f3(D.gamma_min), 'unbounded'], strong: true });
  return (
    <figure style={{ margin: 0 }}>
      <Caption>The Cayley graph of S₃: relation A swaps the first two, relation B cycles all three</Caption>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center' }}>
        <CayleyPanel id="cay-d3" title="D₃: a reflection and a rotation" points={D.points} />
        <CayleyPanel id="cay-transe" title="trained TransE (plane)" points={S3.abelian.TransE['1'].points!} />
        <CayleyPanel id="cay-rotate" title="trained RotatE (k = 1)" points={S3.abelian.RotatE['1'].points!} />
      </div>
      <Key items={[
        { label: 'A facts: x → xa', color: 'var(--color-accent)' },
        { label: 'B facts: x → xb', color: 'var(--color-badge-blue-text)', dash: true },
      ]} />
      <Sub>ε = largest fact distance; γ_min = nearest corruption; Theorem 3: an abelian model has γ_min ≤ 5ε</Sub>
      <Table id="s3" head={['model', 'ε', 'γ_min', 'γ_min / ε']} rows={rows} />
      <Readouts>
        <Readout label="D₃: nearest two entities" value={f3(D.separation)} id="d3-sep" emphasis />
        <Readout label="best abelian ratio" value={f3(Math.max(...runs.map((x) => x.a.ratio)))} id="best-ratio" />
      </Readouts>
      <Note>
        Translations and rotations commute, so whatever the margin, an abelian model keeps the false triple (xa, B, xba)
        within five times its worst true one. The dihedral group&rsquo;s reflection and rotation do not commute, and it fits
        all twelve facts exactly with the six entities apart.
      </Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// D. Link prediction.
// ---------------------------------------------------------------------------------------------------------------------

function LinkView() {
  const [pattern, setPattern] = useState<Pattern>('symmetry');
  const [model, setModel] = useState<Model>('RotatE');
  const qs = QUERIES.filter((q) => q.pattern === pattern);
  const [qi, setQi] = useState(0);
  const q = qs[Math.min(qi, qs.length - 1)];
  const m = q.models[model];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Hidden facts as tail queries: each model&rsquo;s top five filtered candidates</Caption>
      <Row>
        <Label>pattern</Label>
        {PATTERNS.map((p) => (
          <Pill key={p} active={pattern === p} onClick={() => { setPattern(p); setQi(0); }} data={{ 'data-pattern': p }}>{p}</Pill>
        ))}
      </Row>
      <ModelPills model={model} setModel={setModel} />
      <Row>
        <Label>query</Label>
        <select value={Math.min(qi, qs.length - 1)} onChange={(e) => setQi(Number(e.target.value))} data-control="query"
          aria-label="hidden fact" style={{ fontFamily: sans, fontSize: '0.75rem', maxWidth: '100%', minWidth: 0 }}>
          {qs.map((x, i) => <option key={i} value={i}>{`${ENTS[x.h]} ${x.r} ?`}</option>)}
        </select>
      </Row>
      <div data-panel="query" style={{ fontFamily: sans, fontSize: '0.8rem', margin: '0.4rem 0', overflowWrap: 'anywhere' }}>
        ({ENTS[q.h]}, <span style={{ fontFamily: mono }}>{q.r}</span>, ?) — hidden answer: <strong>{ENTS[q.t]}</strong>
      </div>
      <ol data-panel="top" style={{ fontFamily: sans, fontSize: '0.8rem', margin: '0.2rem 0 0.4rem', paddingLeft: '1.4rem' }}>
        {m.top.map((e) => (
          <li key={e} data-candidate={e} style={{ color: e === q.t ? 'var(--color-accent)' : e === q.h ? 'var(--color-text-secondary)' : 'var(--color-text)', fontWeight: e === q.t ? 600 : 400, overflowWrap: 'anywhere' }}>
            {ENTS[e]}{e === q.t ? ' (the hidden answer)' : e === q.h ? ' (the query entity itself)' : ''}
          </li>
        ))}
      </ol>
      <Readouts>
        <Readout label={`${model}: filtered rank of the answer`} value={m.rank.toFixed(1)} id="rank" emphasis />
        <Readout label={`${model}: ${pattern} MRR (3 seeds)`} value={ms(PAT[model][pattern].mrr)} id="pattern-mrr" />
      </Readouts>
      <Sub>filtered MRR on every hidden fact, both directions, mean ± sd over three seeds</Sub>
      <Table id="patterns" head={['model', ...PATTERNS, 'direction']}
        rows={MODELS.map((mm) => ({ key: mm, strong: mm === model, cells: [mm, ...PATTERNS.map((p) => ms(PAT[mm][p].mrr)), f3(PAT[mm].direction[0])] }))} />
      <Note>
        Direction is the share of antisymmetric facts scored above their reversal. DistMult ties every reversal (Proposition 3),
        which filtered MRR never sees. TransE&rsquo;s COMPETES_WITH vector is nearly zero, so its top candidate is often the query
        company itself. Chosen settings: {MODELS.map((mm) => `${mm} ${CONFIGS[mm].gamma ? `γ ${CONFIGS[mm].gamma}` : `λ ${CONFIGS[mm].lam}`}, ${CONFIGS[mm].steps} steps`).join('; ')}.
      </Note>
    </figure>
  );
}

// ---------------------------------------------------------------------------------------------------------------------
// E. Completing extraction.
// ---------------------------------------------------------------------------------------------------------------------

function CompletionView() {
  const [tau, setTau] = useState<'0.5' | '0.1'>('0.5');
  const C = COMP[tau];
  const isEx = (f: Fact) => f.h === EX.h && f.r === EX.r && f.t === EX.t;
  const rels = Object.keys(C.metrics.TransE);
  const fact = (f: Fact) => `(${ENTS[f.h]}, ${f.r}, ${ENTS[f.t]})`;
  const box: CSSProperties = { maxHeight: '14rem', overflowY: 'auto', border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.3rem 0.5rem' };
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Topic 23&rsquo;s facts about six held-out companies, embedded beside the training roster&rsquo;s knowledge base</Caption>
      <Row>
        <Label>extraction threshold</Label>
        <Pill active={tau === '0.5'} onClick={() => setTau('0.5')} data={{ 'data-tau': '0.5' }}>0.5</Pill>
        <Pill active={tau === '0.1'} onClick={() => setTau('0.1')} data={{ 'data-tau': '0.1' }}>0.1</Pill>
      </Row>
      <Readouts>
        <Readout label="known facts" value={String(C.known)} id="known" />
        <Readout label="extracted (true)" value={`${C.extracted.length} (${C.true})`} id="extracted" emphasis />
        <Readout label="missing facts to rank" value={String(C.missing.length)} id="missing" />
      </Readouts>
      <Sub>missing facts, with each model&rsquo;s filtered rank of the tail (seed 0)</Sub>
      <div style={box} data-panel="missing">
        {C.missing.map((f, i) => (
          <div key={i} data-missing={i} style={{ fontFamily: sans, fontSize: '0.75rem', padding: '0.15rem 0', borderBottom: '1px solid var(--color-border)', fontWeight: isEx(f) ? 600 : 400, color: isEx(f) ? 'var(--color-accent)' : 'var(--color-text)' }}>
            <div style={{ overflowWrap: 'anywhere' }}>{fact(f)}{f.score > 0 ? ` — extraction scored ${f3(f.score)}` : ''}</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0 0.8rem', fontFamily: mono, color: 'var(--color-text-secondary)' }}>
              {MODELS.map((m) => <span key={m} data-rank={m}>{m} {f.ranks[m].toFixed(1)}</span>)}
            </div>
          </div>
        ))}
      </div>
      <Sub>MRR and Hits@10 on the missing facts, both directions, mean over three seeds</Sub>
      <Table id="completion" head={['model', ...rels.flatMap((r) => [`${r} MRR`, 'H@10'])]}
        rows={MODELS.map((m) => ({ key: m, cells: [m, ...rels.flatMap((r) => [ms(C.metrics[m][r].mrr), f3(C.metrics[m][r].h10[0])])] }))} />
      <Sub>the extracted facts the models trained on</Sub>
      <div style={box} data-panel="extracted">
        {C.extracted.map((f, i) => (
          <div key={i} data-extracted={i} style={{ fontFamily: sans, fontSize: '0.75rem', display: 'flex', flexWrap: 'wrap', gap: '0 0.6rem', padding: '0.1rem 0' }}>
            <span style={{ fontFamily: mono }}>{f3(f.score)}</span>
            <span style={{ color: f.true ? 'var(--color-text)' : 'var(--color-badge-amber-text)', overflowWrap: 'anywhere', minWidth: 0 }}>
              {fact(f)}{f.true ? '' : ' — false'}
            </span>
          </div>
        ))}
      </div>
      <Note>
        The subsidiary facts follow from extracted PARENT_OF facts by inversion, and three models rank them first. No extracted
        fact links a held-out company to a competitor, so competitors are guessed by sector: the right neighborhood, rarely the
        right neighbor. At 0.1 the extra facts are mostly false, and they cost TransE most.
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'toy', label: 'Translate or rotate' },
  { key: 'phases', label: 'Relation phases' },
  { key: 's3', label: 'Non-commuting' },
  { key: 'link', label: 'Link prediction' },
  { key: 'complete', label: 'Completing extraction' },
] as const;

export default function KnowledgeGraphEmbeddingLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('toy');
  return (
    <div data-lab="knowledge-graph-embedding" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'toy' && <ToyView />}
      {view === 'phases' && <PhasesView />}
      {view === 's3' && <S3View />}
      {view === 'link' && <LinkView />}
      {view === 'complete' && <CompletionView />}
    </div>
  );
}
