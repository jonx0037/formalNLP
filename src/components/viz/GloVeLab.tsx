import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/glove-embedding-geometry/glove_embedding_geometry.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/glove-lab.json';

/**
 * GloVe Laboratory — two linked panels for the glove-embedding-geometry topic.
 *   A. "Three weights, one count": for one of revenue's contexts at the eight-word window, the cell's loss above its
 *      minimum under GloVe f(a)(s − log a)², Poisson a(e^δ − 1 − δ), and SGNS (Thm 1, Cor 1), with the trained scores;
 *      a context never seen with revenue has no GloVe term at all. Bars: each weight's share by count band, GloVe's
 *      recomputed from the count histogram for the chosen x_max and γ.
 *   B. "What the vectors know": two knowledge-base relations as offsets in each factorization of one set of scores,
 *      with 3CosAdd's accuracy and the gauge-free solver's (Thm 3, Cor 2); and test AUC against corpus length for
 *      GloVe, GloVe with imputed zeros (Def 2), and the SVD of the same counts.
 */

type Cell = { context: string; a: number; b: number; log_a: number | null; glove_s: number; sgns_x: number; x_star: number | null };
type Variant = { xy: [number, number][]; open: number; within: number; b_nearest: number; defect: number; gauge_free: number; answers: string[][] };
type Relation = { label: string; words: string[]; pairs: [string, string][]; questions: [string, string, string][]; methods: Record<string, Record<string, Variant>> };
type Scale = { docs: number[]; singleton_share: number[]; glove_test_auc: number[]; imputed_test_auc: number[]; svd_test_auc: number[] };

const W = payload.weights as unknown as {
  row: string; cells: Cell[]; hist: number[]; bands: string[]; share: Record<string, number[]>; rare: Record<string, number>;
};
const ANALOGIES = payload.analogies as unknown as Record<string, Relation>;
const SCALE = payload.scale as unknown as Scale;
const X_MAX0 = payload.x_max as number;
const GAMMA0 = payload.gamma as number;
const K = payload.k as number;
const ALPHA = payload.alpha as number;
const CHANCE_AUC = payload.chance_auc as number;
const RELATIONS = Object.keys(ANALOGIES);

const X_MAXES = [10, 100];
const GAMMAS = [0.5, 0.75, 1];
const VARIANT_LABEL: Record<string, string> = {
  trained: 'as trained', balanced: 'balanced (p = ½)', centered: 'centered (a translation)', random: 'random basis change, condition 100',
};
const METHOD_LABEL: Record<string, string> = { glove: 'GloVe (W + C)', sgns: 'SGNS (W)' };

// One token per count band, as in the word2vec lab; segment order is the band order, so color is never the only cue.
const BAND_COLOR = [
  'var(--color-badge-red-text)', 'var(--color-badge-amber-text)', 'var(--color-badge-green-text)',
  'var(--color-badge-blue-text)', 'var(--color-accent)',
];
// Each objective: a token and a dash pattern.
const SERIES = {
  glove: { color: 'var(--color-accent)', dash: undefined as string | undefined, label: 'GloVe f(a)(s − log a)²' },
  poisson: { color: 'var(--color-accent-secondary)', dash: '6 3', label: 'Poisson a(e^δ − 1 − δ)' },
  sgns: { color: 'var(--color-badge-blue-text)', dash: '2 3', label: 'SGNS −[a log σ + b log σ(−·)]' },
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';

const logSigmoid = (x: number) => (x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)));
const softplus = (x: number) => -logSigmoid(-x);
const gloveWeight = (x: number, xMax: number, gamma: number) => Math.min(1, (x / xMax) ** gamma);
const bandOf = (a: number) => (a <= 1 ? 0 : a <= 4 ? 1 : a <= 19 ? 2 : a <= 99 ? 3 : 4);
const fmtW = (v: number) => (v >= 1 ? v.toFixed(2) : v.toPrecision(3));

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
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)' }}>
        {value}
      </div>
    </div>
  );
}

function Row({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.45rem', ...style }}>{children}</div>;
}

function Note({ children }: { children: ReactNode }) {
  return <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>{children}</p>;
}

// Panel A geometry: δ (observed) or the score s (unobserved) across, loss above its minimum from 0 to 3 up.
const CW = 600, CH = 230, CM = { top: 22, right: 16, bottom: 34, left: 40 };
const D_LO = -3, D_HI = 3, S_LO = -8, S_HI = 4, Y_HI = 3;
// Panel B geometry.
const PW = 600, PH = 330, PM = { top: 18, right: 18, bottom: 18, left: 18 };
const LW = 600, LH = 250, LM = { top: 22, right: 20, bottom: 34, left: 44 };

/** Each weight's share by count band; GloVe's from the histogram of observed counts (1…99, then ≥ 100, where f = 1). */
function gloveShares(xMax: number, gamma: number): number[] {
  const mass = [0, 0, 0, 0, 0];
  W.hist.forEach((n, i) => {
    const a = i + 1;
    mass[i === W.hist.length - 1 ? 4 : bandOf(a)] += n * (i === W.hist.length - 1 ? 1 : gloveWeight(a, xMax, gamma));
  });
  const tot = mass.reduce((s, v) => s + v, 0);
  return mass.map((v) => v / tot);
}

export default function GloVeLab() {
  const [cellIdx, setCellIdx] = useState(0);
  const [xMax, setXMax] = useState(X_MAX0);
  const [gamma, setGamma] = useState(GAMMA0);
  const [view, setView] = useState<'analogies' | 'scale'>('analogies');
  const [relation, setRelation] = useState(RELATIONS[0]);
  const [method, setMethod] = useState('glove');
  const [variant, setVariant] = useState('trained');
  const [qIdx, setQIdx] = useState(0);

  // ----- Panel A -----
  const cell = W.cells[cellIdx];
  const observed = cell.a > 0;
  const fA = observed ? gloveWeight(cell.a, xMax, gamma) : 0;
  const omega = observed ? (cell.a * cell.b) / (cell.a + cell.b) : 0;
  const sgnsLoss = (x: number) => -(cell.a * logSigmoid(x) + cell.b * logSigmoid(-x));
  const xStarExact = observed ? Math.log(cell.a / cell.b) : 0;
  const lo = observed ? D_LO : S_LO, hi = observed ? D_HI : S_HI;

  const curves = useMemo(() => {
    const grid = d3.range(lo, hi + 1e-9, 0.02);
    if (observed) {
      const base = sgnsLoss(xStarExact);
      return {
        glove: grid.map((d) => [d, fA * d * d] as [number, number]),
        poisson: grid.map((d) => [d, cell.a * (Math.expm1(d) - d)] as [number, number]),
        sgns: grid.map((d) => [d, sgnsLoss(xStarExact + d) - base] as [number, number]),
      };
    }
    return {
      glove: grid.map((s) => [s, 0] as [number, number]),
      poisson: grid.map((s) => [s, Math.exp(s)] as [number, number]),
      sgns: grid.map((s) => [s, cell.b * softplus(s)] as [number, number]),
    };
  }, [cellIdx, xMax, gamma]); // eslint-disable-line react-hooks/exhaustive-deps

  const cx = d3.scaleLinear().domain([lo, hi]).range([0, CW - CM.left - CM.right]);
  const cy = d3.scaleLinear().domain([0, Y_HI]).range([CH - CM.top - CM.bottom, 0]).clamp(true);
  const line = d3.line<[number, number]>().defined((p) => p[1] <= Y_HI * 1.05).x((p) => cx(p[0])).y((p) => cy(p[1]));
  const gloveDot = observed ? cell.glove_s - (cell.log_a as number) : cell.glove_s;
  const sgnsDot = observed ? cell.sgns_x - (cell.x_star as number) : cell.sgns_x;
  const yAt = (pts: [number, number][], x: number) => pts.reduce((best, p) => (Math.abs(p[0] - x) < Math.abs(best[0] - x) ? p : best))[1];
  const gloveEmph = method === 'glove' ? 3 : 2;

  const shares: Record<string, number[]> = { ...W.share, glove: gloveShares(xMax, gamma) };
  // Fixed rows read the payload's share on pairs seen ≤ 4 (rounded from the exact sum); GloVe's follows x_max and γ.
  const rare = (key: string) => (key === 'glove' ? shares.glove[0] + shares.glove[1] : W.rare[key]);
  const isDefault = xMax === X_MAX0 && gamma === GAMMA0;

  // ----- Panel B -----
  const rel = ANALOGIES[relation];
  const variants = Object.keys(rel.methods[method]);
  const v = rel.methods[method][variants.includes(variant) ? variant : 'trained'];
  const q = rel.questions[qIdx];
  const expected = rel.pairs.find(([h]) => h === q[2])?.[1] ?? '';
  const answers = v.answers[qIdx];
  const heads = new Set(rel.pairs.map(([h]) => h));
  const pos = new Map(rel.words.map((w, i) => [w, v.xy[i]] as [string, [number, number]]));
  const xs = v.xy.map((p) => p[0]), ys = v.xy.map((p) => p[1]);
  const pad = (lo2: number, hi2: number) => { const r = hi2 - lo2 || 1; return [lo2 - 0.08 * r, hi2 + 0.08 * r]; };
  const px = d3.scaleLinear().domain(pad(d3.min(xs)!, d3.max(xs)!)).range([0, PW - PM.left - PM.right]);
  const py = d3.scaleLinear().domain(pad(d3.min(ys)!, d3.max(ys)!)).range([PH - PM.top - PM.bottom, 0]);
  const inQuestion = (w: string) => w === q[0] || w === q[1] || w === q[2] || w === expected;

  const lx = d3.scaleLog().domain([250, 3600]).range([0, LW - LM.left - LM.right]);
  const ly = d3.scaleLinear().domain([0.4, 1]).range([LH - LM.top - LM.bottom, 0]);
  const scaleSeries: { key: keyof Scale; label: string; color: string; dash?: string; marker: 'circle' | 'square' | 'diamond' }[] = [
    { key: 'glove_test_auc', label: 'GloVe', color: SERIES.glove.color, marker: 'circle' },
    { key: 'imputed_test_auc', label: 'GloVe, unseen pairs as half a count', color: 'var(--color-badge-green-text)', dash: '6 3', marker: 'square' },
    { key: 'svd_test_auc', label: 'SVD of SPPMI₅ (same counts)', color: SERIES.sgns.color, dash: '2 3', marker: 'diamond' },
  ];

  const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;

  return (
    <div data-lab="glove" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Three weights, one count — the loss of one (word, context) cell under GloVe, Poisson, and SGNS
        </figcaption>
        <Row>
          {W.cells.map((c, i) => (
            <Pill key={c.context} active={i === cellIdx} onClick={() => setCellIdx(i)} data={{ 'data-cell': c.context }}>
              {`${W.row} · ${c.context} (seen ${c.a})`}
            </Pill>
          ))}
        </Row>
        <Row>
          {X_MAXES.map((x) => (
            <Pill key={x} active={x === xMax} onClick={() => setXMax(x)} data={{ 'data-xmax': String(x) }}>{`GloVe cap x_max = ${x}`}</Pill>
          ))}
          {GAMMAS.map((g) => (
            <Pill key={g} active={g === gamma} onClick={() => setGamma(g)} data={{ 'data-gamma': String(g) }}>{`exponent γ = ${g}`}</Pill>
          ))}
        </Row>
        <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="weights"
          aria-label={observed
            ? `Loss above its minimum for ${W.row} with ${cell.context}, seen ${cell.a} times, under GloVe, Poisson, and SGNS, against the distance from each objective's optimum`
            : `Loss for ${W.row} with ${cell.context}, never seen together: GloVe has no term, Poisson and SGNS push the score down`}>
          <g transform={`translate(${CM.left},${CM.top})`}>
            {[0, 1, 2, 3].map((t) => (
              <g key={t}>
                <line x1={0} x2={CW - CM.left - CM.right} y1={cy(t)} y2={cy(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                <text x={-6} y={cy(t) + 3} textAnchor="end" style={axisText}>{t}</text>
              </g>
            ))}
            {d3.range(lo, hi + 1, observed ? 1 : 2).map((t) => (
              <text key={t} x={cx(t)} y={CH - CM.top - CM.bottom + 14} textAnchor="middle" style={axisText}>{t}</text>
            ))}
            {(['poisson', 'sgns', 'glove'] as const).map((k) => (
              <path key={k} d={line(curves[k])!} data-series={k}
                style={{ fill: 'none', stroke: SERIES[k].color, strokeWidth: k === 'glove' ? gloveEmph : 2, strokeDasharray: SERIES[k].dash }} />
            ))}
            <circle cx={cx(gloveDot)} cy={cy(observed ? fA * gloveDot * gloveDot : 0)} r={5} data-series="glove-trained"
              style={{ fill: SERIES.glove.color, stroke: 'var(--color-text)', strokeWidth: 1 }} />
            <rect x={cx(sgnsDot) - 4.5} y={cy(yAt(curves.sgns, sgnsDot)) - 4.5} width={9} height={9} data-series="sgns-trained"
              style={{ fill: 'none', stroke: SERIES.sgns.color, strokeWidth: 2 }} />
          </g>
          <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
            {observed ? `loss above its minimum · ${W.row} · ${cell.context}, seen ${cell.a} times` : `${W.row} · ${cell.context}: never seen together`}
          </text>
          <text x={CW / 2} y={CH - 4} textAnchor="middle" style={axisText}>
            {observed ? 'δ = distance of the score from that objective’s optimum' : 'score s (no optimum exists)'}
          </text>
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
          {(['glove', 'poisson', 'sgns'] as const).map((k) => (
            <span key={k} data-legend={k} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
              <svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" style={{ stroke: SERIES[k].color, strokeWidth: 2, strokeDasharray: SERIES[k].dash }} /></svg>
              {SERIES[k].label}
            </span>
          ))}
          <span>· dot: trained GloVe · square: trained SGNS</span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>
          <Readout id="a" label="times seen, a" value={String(cell.a)} />
          <Readout id="w-glove" label="GloVe weight f(a)" value={observed ? fmtW(fA) : 'no term'} emphasis />
          <Readout id="w-poisson" label="Poisson weight a" value={observed ? String(cell.a) : '—'} />
          <Readout id="w-sgns" label={`SGNS weight ab/(a + b), b = ${cell.b.toFixed(2)}`} value={observed ? fmtW(omega) : '—'} />
          <Readout id="glove-s" label="trained GloVe score s (log a)" value={observed ? `${cell.glove_s.toFixed(3)} (${(cell.log_a as number).toFixed(3)})` : `${cell.glove_s.toFixed(3)} (free)`} />
          <Readout id="sgns-x" label="trained SGNS score (x*)" value={observed ? `${cell.sgns_x.toFixed(3)} (${(cell.x_star as number).toFixed(3)})` : `${cell.sgns_x.toFixed(3)} (none)`} />
        </div>
        <div data-panel="shares" style={{ marginTop: '0.75rem', fontFamily: sans, fontSize: '0.78rem' }}>
          <div style={{ color: 'var(--color-text-secondary)', marginBottom: '0.3rem' }}>
            Each weight’s total, split by how often the pair was seen (all {W.hist.reduce((s, n) => s + n, 0).toLocaleString('en-US')} observed cells at the eight-word window)
          </div>
          {([['cells', 'share of cells'], ['glove', 'GloVe f(X)'], ['poisson', 'Poisson X'], ['sgns', `SGNS ab/(a + b), k = ${K}, α = ${ALPHA}`]] as const).map(([key, label]) => (
            <div key={key} data-share-row={key} style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.2rem 0.6rem', marginBottom: '0.3rem' }}>
              <span style={{ flex: '0 1 12rem', minWidth: 0 }}>{label}</span>
              <span style={{ flex: '1 1 10rem', minWidth: 0, display: 'flex', height: '0.75rem', borderRadius: '0.2rem', overflow: 'hidden' }}>
                {shares[key].map((s, i) => (
                  <span key={i} title={`seen ${W.bands[i]}: ${(100 * s).toFixed(1)}%`} style={{ width: `${100 * s}%`, background: BAND_COLOR[i], opacity: 0.85 }} />
                ))}
              </span>
              <span data-readout={`rare-${key}`} style={{ fontFamily: mono, fontSize: '0.74rem' }}>{`seen ≤ 4: ${rare(key).toFixed(3)}`}</span>
            </div>
          ))}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.75rem', fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
            <span>segments, left to right — times seen:</span>
            {W.bands.map((name, i) => (
              <span key={name} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                <span aria-hidden="true" style={{ width: '0.7rem', height: '0.7rem', background: BAND_COLOR[i], display: 'inline-block', borderRadius: '0.15rem' }} />
                {name}
              </span>
            ))}
          </div>
        </div>
        <Note>
          Curvature is weight: each curve’s width at its minimum is set by that objective’s weight on this cell. GloVe caps its
          weight at 1 and tempers it below x_max, so it spends most of its weight on pairs seen once or a few times
          {isDefault ? '' : ' (the shares above follow your x_max and γ)'}, whose log counts are 0 or close to it. For a context
          never seen with {W.row}, Poisson and SGNS keep pushing the score down; GloVe has no term at all, and its score is free.
        </Note>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · What the vectors know — relations as offsets, and what more text buys
        </figcaption>
        <Row>
          <Pill active={view === 'analogies'} onClick={() => setView('analogies')} data={{ 'data-view': 'analogies' }}>analogies and the gauge</Pill>
          <Pill active={view === 'scale'} onClick={() => setView('scale')} data={{ 'data-view': 'scale' }}>more text, and unseen pairs</Pill>
        </Row>

        {view === 'analogies' ? (
          <>
            <Row>
              {RELATIONS.map((r) => (
                <Pill key={r} active={r === relation} onClick={() => { setRelation(r); setQIdx(0); }} data={{ 'data-relation': r }}>{ANALOGIES[r].label}</Pill>
              ))}
              {Object.keys(METHOD_LABEL).map((m) => (
                <Pill key={m} active={m === method} onClick={() => { setMethod(m); if (!Object.keys(rel.methods[m]).includes(variant)) setVariant('trained'); }}
                  data={{ 'data-method': m }}>{METHOD_LABEL[m]}</Pill>
              ))}
            </Row>
            <Row>
              {variants.map((n) => (
                <Pill key={n} active={n === (variants.includes(variant) ? variant : 'trained')} onClick={() => setVariant(n)} data={{ 'data-variant': n }}>{VARIANT_LABEL[n]}</Pill>
              ))}
            </Row>
            <Row>
              {rel.questions.map((qq, i) => (
                <Pill key={qq.join()} active={i === qIdx} onClick={() => setQIdx(i)} data={{ 'data-question': String(i) }}>{`${qq[0]} : ${qq[1]} :: ${qq[2]} : ?`}</Pill>
              ))}
            </Row>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="analogies"
              aria-label={`The ${rel.label} words of ${METHOD_LABEL[method]}, ${VARIANT_LABEL[variant] ?? 'as trained'}, in their own two leading principal directions, with an arrow from each head to its tail`}>
              <defs>
                <marker id="glove-lab-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M0,0 L8,4 L0,8 z" style={{ fill: 'var(--color-text-secondary)' }} />
                </marker>
              </defs>
              <g transform={`translate(${PM.left},${PM.top})`}>
                {rel.pairs.map(([h, t]) => {
                  const a = pos.get(h)!, b = pos.get(t)!;
                  const hot = h === q[0] || h === q[2];
                  return (
                    <line key={`${h}-${t}`} x1={px(a[0])} y1={py(a[1])} x2={px(b[0])} y2={py(b[1])} markerEnd="url(#glove-lab-arrow)" data-arrow={h}
                      style={{ stroke: hot ? 'var(--color-accent)' : 'var(--color-text-secondary)', strokeWidth: hot ? 1.8 : 0.8, opacity: hot ? 1 : 0.45 }} />
                  );
                })}
                {rel.words.map((w) => {
                  const p = pos.get(w)!;
                  const head = heads.has(w);
                  const hot = inQuestion(w);
                  return (
                    <g key={w} data-word={w}>
                      {head
                        ? <circle cx={px(p[0])} cy={py(p[1])} r={hot ? 4.5 : 3} style={{ fill: 'var(--color-text-secondary)' }} />
                        : <rect x={px(p[0]) - (hot ? 4.5 : 3)} y={py(p[1]) - (hot ? 4.5 : 3)} width={hot ? 9 : 6} height={hot ? 9 : 6} style={{ fill: 'var(--color-accent)' }} />}
                      <text x={px(p[0]) + 6} y={py(p[1]) - 5}
                        style={{ fontFamily: sans, fontSize: hot ? '12px' : '9px', fontWeight: hot ? 700 : 400, fill: hot ? 'var(--color-text)' : 'var(--color-text-secondary)' }}>
                        {w}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
            <div style={{ fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              circles: heads · squares: tails · arrows: head → tail · the question’s words are bold
            </div>
            <div data-panel="answers" style={{ fontFamily: sans, fontSize: '0.82rem', margin: '0.5rem 0', overflowWrap: 'anywhere' }}>
              <span>{`${q[0]} : ${q[1]} :: ${q[2]} : ?  3CosAdd's top three — `}</span>
              <span data-readout="answers" style={{ fontFamily: mono }}>
                {answers.map((w, i) => (i === 0 ? `${w}${w === expected ? ' ✓' : ' ✗'}` : w)).join(' · ')}
              </span>
              <span style={{ color: 'var(--color-text-secondary)' }}>{` (expected ${expected})`}</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.25rem' }}>
              <Readout id="open" label={`3CosAdd, whole vocabulary (${rel.label})`} value={v.open.toFixed(3)} emphasis />
              <Readout id="within" label="3CosAdd, among the tails" value={v.within.toFixed(3)} />
              <Readout id="b-nearest" label="b itself nearest, before exclusion" value={v.b_nearest.toFixed(3)} />
              <Readout id="gauge-free" label="gauge-free solver (any variant)" value={v.gauge_free.toFixed(3)} emphasis />
              <Readout id="defect" label="median parallelogram defect" value={v.defect.toFixed(3)} />
            </div>
            <Note>
              Every variant of a method has the same scores, so the same loss: the variants differ only by a change of basis (and,
              for GloVe, a translation its biases absorb). 3CosAdd reads cosines between word vectors, so its answers move. The
              gauge-free solver picks the tail whose score rows come closest to a parallelogram, and the defect measures how far
              they are; both read the scores alone, so neither moves. No model is near an exact parallelogram (defect near 1),
              yet the scores still point to the right tail far more often than chance.
            </Note>
          </>
        ) : (
          <>
            <svg viewBox={`0 0 ${LW} ${LH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="scale"
              aria-label="Test AUC on the planted-sector benchmark against corpus length, for GloVe, GloVe with unseen pairs imputed at half a count, and the SVD of the same counts">
              <g transform={`translate(${LM.left},${LM.top})`}>
                {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => (
                  <g key={t}>
                    <line x1={0} x2={LW - LM.left - LM.right} y1={ly(t)} y2={ly(t)}
                      style={{ stroke: 'var(--color-border)', strokeWidth: t === CHANCE_AUC ? 1.2 : 0.5, strokeDasharray: t === CHANCE_AUC ? '4 3' : undefined }} />
                    <text x={-6} y={ly(t) + 3} textAnchor="end" style={axisText}>{t.toFixed(1)}</text>
                  </g>
                ))}
                <text x={LW - LM.left - LM.right} y={ly(CHANCE_AUC) - 4} textAnchor="end" style={axisText}>chance</text>
                {SCALE.docs.map((n) => (
                  <text key={n} x={lx(n)} y={LH - LM.top - LM.bottom + 14} textAnchor="middle" style={axisText}>{n.toLocaleString('en-US')}</text>
                ))}
                {scaleSeries.map((s) => {
                  const vals = SCALE[s.key] as number[];
                  const pts = SCALE.docs.map((n, i) => [lx(n), ly(vals[i])] as [number, number]);
                  return (
                    <g key={s.key} data-series={s.key}>
                      <path d={d3.line()(pts)!} style={{ fill: 'none', stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />
                      {pts.map(([x, y], i) => (
                        s.marker === 'circle' ? <circle key={i} cx={x} cy={y} r={4} style={{ fill: s.color }} />
                          : s.marker === 'square' ? <rect key={i} x={x - 4} y={y - 4} width={8} height={8} style={{ fill: s.color }} />
                            : <path key={i} d={`M${x},${y - 5} L${x + 5},${y} L${x},${y + 5} L${x - 5},${y} Z`} style={{ fill: s.color }} />
                      ))}
                    </g>
                  );
                })}
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>test AUC, planted sectors (115 nouns)</text>
              <text x={LW / 2} y={LH - 4} textAnchor="middle" style={axisText}>documents of text (the same generator, continued)</text>
            </svg>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              {scaleSeries.map((s) => (
                <span key={s.key} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                  <svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} /></svg>
                  {s.label}
                </span>
              ))}
            </div>
            <div data-panel="scale-table" style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>
              {SCALE.docs.map((n, i) => (
                <Readout key={n} id={`scale-${n}`} label={`${n.toLocaleString('en-US')} docs: seen once · GloVe · imputed · SVD`}
                  value={`${SCALE.singleton_share[i].toFixed(3)} · ${SCALE.glove_test_auc[i].toFixed(3)} · ${SCALE.imputed_test_auc[i].toFixed(3)} · ${SCALE.svd_test_auc[i].toFixed(3)}`} />
              ))}
            </div>
            <Note>
              The generator, run on, keeps its sector vocabularies, so the same benchmark nouns keep their sectors. As the text grows,
              fewer observed pairs are seen only once, and GloVe’s vectors begin to recover the sectors. Counting every unseen pair as
              seen half a time, with GloVe’s own weight, lets the loss see which pairs never occur, and recovers them much faster.
              Both GloVe curves are trained by alternating least squares.
            </Note>
          </>
        )}
      </figure>
    </div>
  );
}
