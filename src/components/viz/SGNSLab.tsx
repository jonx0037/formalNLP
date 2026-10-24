import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/word2vec-implicit-factorization/word2vec_implicit_factorization.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/sgns-lab.json';

/**
 * SGNS Laboratory — two linked panels for the word2vec-implicit-factorization topic.
 *   A. "One cell of the objective": for one of revenue's contexts at the eight-word window, the cell's loss
 *      −[a log σ(x) + b log σ(−x)] above its minimum, with its optimum x* = PMI_α − log k (Thm 1), its local weight
 *      ab/(a + b) (Thm 2, Cor 1), and the trained model's score. An unobserved context has no minimum.
 *   B. "Training and the gauge": the controlled model's scores against their targets x* on 200 sampled pairs at each
 *      checkpoint, by count band (Cor 1), with the unobserved cells pushed down; and five factorizations of the one
 *      trained product, all with the same loss, with their benchmark scores and neighbors (Prop 3, Cor 2).
 */

type Pair = {
  context: string; a: number; n_w: number; N: number; trained: number;
  p_noise: Record<string, number>; x_star: Record<string, number | null>; weight: Record<string, number | null>;
};
type Checkpoint = {
  epoch: number; x: number[]; x_unobs: number[]; loss: number; imbalance: number; band_r: number[]; gauge_sv: number[];
};
type Fit = { bands: string[]; band: number[]; x_star: number[]; pairs: string[]; checkpoints: Checkpoint[]; floor: number };
type Variant = { test_auc: number; test_nn: number; neighbors: Record<string, [string, number][]> };

const ROW = payload.row as string;
const PAIRS = payload.pairs as unknown as Pair[];
const KS = payload.ks as number[];
const ALPHAS = payload.alphas as string[];
const K_TRAINED = payload.k as number;
const ALPHA_TRAINED = String(payload.alpha);
const FIT = payload.fit as unknown as Fit;
const GAUGE = payload.gauge as unknown as Record<string, Variant>;
const KAPPA = payload.kappa as number;
const SVD_AUC = payload.svd_test_auc as number;
const CHANCE = payload.chance_nn as number;
const D = payload.d as number;

const VARIANTS: { key: string; label: string }[] = [
  { key: 'trained', label: 'as trained' },
  { key: 'p0', label: 'UΣ⁰ (p = 0)' },
  { key: 'p½', label: 'UΣ^½ (p = ½, balanced)' },
  { key: 'p1', label: 'UΣ (p = 1)' },
  { key: 'random', label: `random A, condition ${KAPPA}` },
];
const GAUGE_WORDS = Object.keys(GAUGE.trained.neighbors);

// One token per count band; point radius grows with the band too, so color is never the only cue.
const BAND_COLOR = [
  'var(--color-badge-red-text)', 'var(--color-badge-amber-text)', 'var(--color-badge-green-text)',
  'var(--color-badge-blue-text)', 'var(--color-accent)',
];
const BAND_RADIUS = [2, 2.4, 2.8, 3.2, 3.6];

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';

const logSigmoid = (x: number) => (x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)));
const bandOf = (a: number) => (a <= 1 ? 0 : a <= 4 ? 1 : a <= 19 ? 2 : a <= 99 ? 3 : 4);
const fmt = (v: number | null | undefined, nd = 3) => (v == null ? '—' : v.toFixed(nd));

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

// Panel A geometry: x from −8 to 6, loss above its minimum from 0 to 3.
const CW = 600, CH = 230, CM = { top: 20, right: 16, bottom: 34, left: 40 };
const X_LO = -8, X_HI = 6, Y_HI = 3;
// Panel B geometry: an unobserved strip on the left, then the scatter.
const FW = 600, FH = 300, FM = { top: 22, right: 16, bottom: 36, left: 40 }, STRIP = 46;
const T_LO = -3.5, T_HI = 4.5, S_LO = -10.5, S_HI = 5;

export default function SGNSLab() {
  const [pairIdx, setPairIdx] = useState(0);
  const [k, setK] = useState(K_TRAINED);
  const [alpha, setAlpha] = useState(ALPHA_TRAINED);
  const [view, setView] = useState<'fit' | 'gauge'>('fit');
  const [ckpt, setCkpt] = useState(FIT.checkpoints.length - 1);
  const [gaugeWord, setGaugeWord] = useState(GAUGE_WORDS[0]);

  const pair = PAIRS[pairIdx];
  const key = `${alpha}|${k}`;
  const observed = pair.a > 0;
  const b = k * pair.n_w * pair.p_noise[alpha];
  const xStar = pair.x_star[key];
  const isTrainedSetting = k === K_TRAINED && alpha === ALPHA_TRAINED;

  // Panel A: the cell's loss above its minimum, computed from the payload's a, #(w), and P_α(c).
  const curve = useMemo(() => {
    const loss = (x: number) => -(pair.a * logSigmoid(x) + b * logSigmoid(-x));
    const base = observed ? loss(Math.log(pair.a / b)) : 0;
    return d3.range(X_LO, X_HI + 1e-9, 0.02).map((x) => [x, loss(x) - base] as [number, number]);
  }, [pair, b, observed]);
  const cx = d3.scaleLinear().domain([X_LO, X_HI]).range([0, CW - CM.left - CM.right]);
  const cy = d3.scaleLinear().domain([0, Y_HI]).range([CH - CM.top - CM.bottom, 0]).clamp(true);
  const curveLine = d3.line<[number, number]>().defined((p) => p[1] <= Y_HI * 1.05).x((p) => cx(p[0])).y((p) => cy(p[1]));

  // Panel B, fit view.
  const cp = FIT.checkpoints[ckpt];
  const fx = d3.scaleLinear().domain([T_LO, T_HI]).range([STRIP, FW - FM.left - FM.right]);
  const fy = d3.scaleLinear().domain([S_LO, S_HI]).range([FH - FM.top - FM.bottom, 0]).clamp(true);
  const pickedName = `${ROW}·${pair.context}`;
  const pickedInSample = FIT.pairs.indexOf(pickedName);
  const pickedBand = observed ? bandOf(pair.a) : -1;
  const emphasis = (band: number, i: number) => (pickedInSample >= 0 ? (i === pickedInSample ? 1 : 0.35) : band === pickedBand ? 1 : 0.3);

  const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;

  return (
    <div data-lab="sgns" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · One cell of the objective — the loss of one (word, context) pair as a function of its score
        </figcaption>
        <Row>
          {PAIRS.map((p, i) => (
            <Pill key={p.context} active={i === pairIdx} onClick={() => setPairIdx(i)} data={{ 'data-pair': p.context }}>
              {`${ROW} · ${p.context} (seen ${p.a})`}
            </Pill>
          ))}
        </Row>
        <Row>
          {KS.map((v) => (
            <Pill key={v} active={v === k} onClick={() => setK(v)} data={{ 'data-k': String(v) }}>{`k = ${v} negatives`}</Pill>
          ))}
          {ALPHAS.map((v) => (
            <Pill key={v} active={v === alpha} onClick={() => setAlpha(v)} data={{ 'data-alpha': v }}>{`noise exponent α = ${Number(v)}`}</Pill>
          ))}
        </Row>
        <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="cell"
          aria-label={observed
            ? `Loss of the pair ${pickedName} above its minimum, against the score x; the minimum is at x* = ${fmt(xStar)}`
            : `Loss of the unobserved pair ${pickedName} against the score x; it decreases without bound as x falls`}>
          <g transform={`translate(${CM.left},${CM.top})`}>
            {[0, 1, 2, 3].map((v) => (
              <g key={v}>
                <line x1={0} x2={CW - CM.left - CM.right} y1={cy(v)} y2={cy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                <text x={-6} y={cy(v) + 3} textAnchor="end" style={axisText}>{v}</text>
              </g>
            ))}
            {d3.range(X_LO, X_HI + 1, 2).map((v) => (
              <text key={v} x={cx(v)} y={CH - CM.top - CM.bottom + 14} textAnchor="middle" style={axisText}>{v}</text>
            ))}
            <path d={curveLine(curve)!} data-series="cell-loss" style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
            {observed && xStar != null && (
              <line x1={cx(xStar)} x2={cx(xStar)} y1={0} y2={CH - CM.top - CM.bottom} data-series="x-star"
                style={{ stroke: 'var(--color-accent-secondary)', strokeWidth: 1.5, strokeDasharray: '5 3' }} />
            )}
            <circle cx={cx(pair.trained)} cy={cy(curve.reduce((best, p) => (Math.abs(p[0] - pair.trained) < Math.abs(best[0] - pair.trained) ? p : best))[1])}
              r={5} data-series="trained" style={{ fill: isTrainedSetting ? 'var(--color-text)' : 'none', stroke: 'var(--color-text)', strokeWidth: 1.5 }} />
          </g>
          <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
            {`loss above its minimum, ${pickedName}${observed ? ' · dashed: x* = PMI_α − log k' : ' · never observed'}`}
          </text>
          <text x={CW / 2} y={CH - 4} textAnchor="middle" style={axisText}>score x = ⟨word vector, context vector⟩</text>
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.25rem' }}>
          <Readout id="a" label="times seen, a" value={String(pair.a)} />
          <Readout id="b" label="expected negative draws, b" value={b.toFixed(2)} />
          <Readout id="x-star" label="optimum x* = PMI_α − log k" value={observed ? fmt(xStar) : 'no minimum'} emphasis />
          <Readout id="weight" label="local weight ab/(a + b)" value={observed ? fmt(pair.weight[key]) : '—'} />
          <Readout id="trained" label={`trained score (k = ${K_TRAINED}, α = ${Number(ALPHA_TRAINED)})`} value={fmt(pair.trained)} />
        </div>
        <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
          Changing k slides the minimum by −log k; the curve's width falls as the weight grows, so frequent pairs are pinned
          and rare ones are not. The dot is the score the trained model learned (filled at the setting it was trained with):
          close to x* for a pair seen dozens of times, far from it for a pair seen once. A context never seen with {ROW} has
          no minimum, only a push toward −∞.
        </p>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Training and the gauge — what SGD fits, and the factorization it picks
        </figcaption>
        <Row>
          <Pill active={view === 'fit'} onClick={() => setView('fit')} data={{ 'data-view': 'fit' }}>fit during training</Pill>
          <Pill active={view === 'gauge'} onClick={() => setView('gauge')} data={{ 'data-view': 'gauge' }}>same loss, five factorizations</Pill>
        </Row>

        {view === 'fit' ? (
          <>
            <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem', fontFamily: sans, fontSize: '0.85rem', marginBottom: '0.25rem' }}>
              <span>epoch <strong data-readout="epoch">{cp.epoch}</strong></span>
              <input type="range" min={0} max={FIT.checkpoints.length - 1} step={1} value={ckpt} aria-label="Training checkpoint"
                onChange={(e) => setCkpt(parseInt(e.target.value, 10))} style={{ flex: '1 1 8rem', minWidth: 0, accentColor: 'var(--color-accent)' }} />
            </label>
            <svg viewBox={`0 0 ${FW} ${FH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="fit"
              aria-label={`Learned score against its target PMI minus log k for ${FIT.x_star.length} sampled pairs at epoch ${cp.epoch}, colored by how often the pair was seen, with unobserved cells in a strip on the left`}>
              <g transform={`translate(${FM.left},${FM.top})`}>
                {[-10, -8, -6, -4, -2, 0, 2, 4].map((v) => (
                  <g key={v}>
                    <line x1={STRIP} x2={FW - FM.left - FM.right} y1={fy(v)} y2={fy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={fy(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {[-2, 0, 2, 4].map((v) => (
                  <text key={v} x={fx(v)} y={FH - FM.top - FM.bottom + 14} textAnchor="middle" style={axisText}>{v}</text>
                ))}
                <line x1={fx(T_LO)} y1={fy(T_LO)} x2={fx(T_HI)} y2={fy(T_HI)} data-series="diagonal"
                  style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '4 4' }} />
                <line x1={STRIP - 4} x2={STRIP - 4} y1={0} y2={FH - FM.top - FM.bottom} style={{ stroke: 'var(--color-border)' }} />
                <text x={STRIP / 2 - 2} y={FH - FM.top - FM.bottom + 14} textAnchor="middle" style={axisText}>unseen</text>
                <g data-series="unobserved" style={{ opacity: !observed ? 1 : 0.55 }}>
                  {cp.x_unobs.map((v, i) => (
                    <circle key={i} cx={6 + ((i * 7) % (STRIP - 14))} cy={fy(v)} r={2} style={{ fill: 'var(--color-text-secondary)' }} />
                  ))}
                </g>
                <g data-series="observed">
                  {cp.x.map((v, i) => {
                    const band = FIT.band[i];
                    return (
                      <circle key={i} cx={fx(FIT.x_star[i])} cy={fy(v)} r={i === pickedInSample ? 6 : BAND_RADIUS[band]} data-band={band}
                        style={{ fill: BAND_COLOR[band], opacity: emphasis(band, i), stroke: i === pickedInSample ? 'var(--color-text)' : 'none', strokeWidth: 1.5 }}>
                        <title>{`${FIT.pairs[i]}: x* = ${FIT.x_star[i]}, learned ${v}`}</title>
                      </circle>
                    );
                  })}
                </g>
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
                {`learned score ⟨w, c⟩ (vertical) against its target PMI_α − log k (horizontal) · window 8, d = ${D}`}
              </text>
              <text x={FW / 2 + STRIP / 2} y={FH - 4} textAnchor="middle" style={axisText}>target x* = PMI_α − log k</text>
            </svg>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.75rem', alignItems: 'center', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              <span>times seen:</span>
              {FIT.bands.map((name, i) => (
                <span key={name} data-legend-band={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                  <svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r={BAND_RADIUS[i] + 0.4} style={{ fill: BAND_COLOR[i] }} /></svg>
                  {`${name}${cp.band_r.length ? ` (r = ${cp.band_r[i].toFixed(3)})` : ''}`}
                </span>
              ))}
              <span>· dashed: learned = target</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.4rem' }}>
              <Readout id="loss" label="loss per pair (floor)" value={`${cp.loss.toFixed(4)} (${FIT.floor.toFixed(4)})`} emphasis />
              <Readout id="r-once" label="r, pairs seen once" value={cp.band_r.length ? cp.band_r[0].toFixed(3) : '—'} />
              <Readout id="r-frequent" label="r, pairs seen 100+" value={cp.band_r.length ? cp.band_r[cp.band_r.length - 1].toFixed(3) : '—'} />
              <Readout id="gauge-sv" label="gauge A's singular values" value={cp.gauge_sv.length ? `${cp.gauge_sv[0].toFixed(3)} – ${cp.gauge_sv[1].toFixed(3)}` : '—'} />
              <Readout id="imbalance" label="‖WᵀW − CᵀC‖ / ‖WᵀW‖" value={cp.imbalance.toFixed(3)} />
            </div>
            <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
              {pickedInSample >= 0
                ? `The ringed point is ${pickedName}, the pair chosen in panel A. `
                : observed ? `${pickedName} is not among the sampled pairs; its count band is highlighted. ` : `${pickedName} is unobserved; the strip on the left holds ${cp.x_unobs.length} unobserved cells. `}
              At epoch 0 every score is 0, since the context vectors start at zero. Frequent pairs reach the diagonal first;
              pairs seen once stay scattered, as their small weight predicts, and unobserved cells sink. The imbalance grows
              well above zero, because SGD is not gradient flow, but the gauge stays near orthogonal from the first epoch on.
            </p>
          </>
        ) : (
          <>
            <Row>
              {GAUGE_WORDS.map((w) => (
                <Pill key={w} active={w === gaugeWord} onClick={() => setGaugeWord(w)} data={{ 'data-word': w }}>{`neighbors of ${w}`}</Pill>
              ))}
            </Row>
            <div data-panel="gauge" style={{ fontFamily: sans, fontSize: '0.8rem' }}>
              {VARIANTS.map(({ key: vk, label }) => {
                const v = GAUGE[vk];
                return (
                  <div key={vk} data-variant={vk} style={{ padding: '0.35rem 0', borderBottom: '1px solid var(--color-border)', minWidth: 0 }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.25rem 0.75rem' }}>
                      <span style={{ flex: '0 1 11rem', minWidth: 0, fontWeight: vk === 'p½' ? 700 : 500 }}>{label}</span>
                      <span style={{ flex: '1 1 5rem', minWidth: '3rem', height: '0.6rem', background: 'var(--color-muted-bg)', borderRadius: '0.2rem', overflow: 'hidden' }}>
                        <span style={{ display: 'block', height: '100%', width: `${Math.max(2, ((v.test_auc - 0.5) / 0.5) * 100)}%`, background: 'var(--color-accent)', opacity: 0.85 }} />
                      </span>
                      <span data-readout={`auc-${vk}`} style={{ fontFamily: mono, fontSize: '0.75rem', whiteSpace: 'nowrap' }}>{`AUC ${v.test_auc.toFixed(3)} · 1-NN ${v.test_nn.toFixed(3)}`}</span>
                    </div>
                    <div data-neighbors={vk} style={{ fontFamily: mono, fontSize: '0.74rem', color: 'var(--color-text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: '0.15rem' }}
                      title={v.neighbors[gaugeWord].map(([n]) => n).join(', ')}>
                      {v.neighbors[gaugeWord].map(([n, c]) => `${n} ${c.toFixed(2)}`).join(' · ')}
                    </div>
                  </div>
                );
              })}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>
              <Readout id="same-loss" label="product WCᵀ, and so the loss" value="identical in all five" emphasis />
              <Readout id="svd-auc" label="SVD of the same matrix, test AUC" value={SVD_AUC.toFixed(3)} />
              <Readout id="chance" label="chance, nearest neighbor" value={CHANCE.toFixed(3)} />
            </div>
            <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
              Each row is the same trained product refactored as (WA, CA⁻ᵀ): the eigenvalue weights p = 0, ½, 1 of the
              distributional-semantics topic, and an arbitrary A. Scores are the sector-ranking AUC and nearest-neighbor accuracy
              on the 115 test nouns. The objective cannot tell these apart; training's near balance picks a factorization close to
              p = ½, the best of the five.
            </p>
          </>
        )}
      </figure>
    </div>
  );
}
