import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/positional-encodings-geometry/positional_encodings_geometry.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the views make. Change the .py, regenerate, never the reverse.
import payload from './data/positional-encoding-lab.json';

/**
 * Positional Encoding Laboratory — two figures for the positional-encodings-geometry topic.
 *   A. "Clocks and kernels": the sinusoidal code as d/2 clock hands at positions m and n (Def 2, Prop 2), with the
 *      normalized kernel (2/d)κ_d(n − m) read from the payload's table (Thm 1 (i)); the kernel against distance for
 *      d = 16, 64, 512 beside its Ci limit and the logarithmic ruler (Thm 1 (ii)); and the trained sinusoidal head's
 *      position–position score at a fixed offset as the position slides, against its commutant projection (Thm 2).
 *   B. "Past the training window": the tournament's accuracies (mean of three seeds) and each scheme's learned score
 *      for a cue at offset r (seed 0); the wavelength census against the training length with the PI, NTK-aware, and
 *      YaRN per-plane scales (Thm 5); and one held-out filing with every reader's answer at every position, and the
 *      accuracies by gap band with ALiBi's horizon (Thm 3 (iii)).
 */

type Lock = { additive: number[]; projected: number };
type Scales = { pi: number[]; ntk: number[]; yarn: number[] };

const CLOCKS = payload.clocks as unknown as { bases: number[]; dims: number[]; kernel: Record<string, number[]> };
const RULER = payload.ruler as unknown as {
  deltas: number[]; kernel: Record<string, number[]>; limit: number[]; log_ruler: number[];
};
const REL = payload.relativity as unknown as { defect: number; lock: Record<string, Lock> };
const TOUR = payload.tournament as unknown as {
  lengths: number[]; accuracy: Record<string, (number | null)[]>; rescues: Record<string, number[]>;
  offset_score: Record<string, number[]>;
};
const CENSUS = payload.census as unknown as { L: number; wavelengths: number[]; complete: boolean[]; scales: Record<string, Scales> };
const FILING = payload.filing as unknown as {
  index: number; tokens: string[]; labels: number[]; truth: number[]; answers: Record<string, number[]>;
  bands: (number | null)[][]; band_accuracy: Record<string, number[]>; horizon: number;
};

const SCHEMES = ['nope', 'sinusoidal', 'learned', 'rope', 'alibi'] as const;
const RESCUES = ['pi', 'ntk', 'yarn'] as const;
type Scheme = (typeof SCHEMES)[number];
type Rescue = (typeof RESCUES)[number];
const STYLE: Record<string, { label: string; color: string; dash?: string }> = {
  nope: { label: 'NoPE', color: 'var(--color-text-secondary)', dash: '2 3' },
  sinusoidal: { label: 'sinusoidal (absolute)', color: 'var(--color-badge-blue-text)' },
  learned: { label: 'learned table (absolute)', color: 'var(--color-badge-green-text)' },
  rope: { label: 'RoPE', color: 'var(--color-accent)' },
  alibi: { label: 'ALiBi', color: 'var(--color-accent-secondary)' },
  'rope+pi': { label: 'RoPE + PI', color: 'var(--color-accent)', dash: '7 3' },
  'rope+ntk': { label: 'RoPE + NTK-aware', color: 'var(--color-accent)', dash: '2 2' },
  'rope+yarn': { label: 'RoPE + YaRN', color: 'var(--color-accent)', dash: '9 3 2 3' },
};
const RESCUE_LABEL: Record<Rescue | 'none', string> = { none: 'no rescaling', pi: 'PI', ntk: 'NTK-aware', yarn: 'YaRN' };
const READERS = Object.keys(FILING.answers);
const BAND_SERIES = Object.keys(FILING.band_accuracy);
const COMPANY_COLORS = ['var(--color-accent)', 'var(--color-badge-blue-text)', 'var(--color-accent-secondary)',
  'var(--color-badge-green-text)', 'var(--color-badge-red-text)'];
const LETTERS = 'ABCDEFGH';
const CENSUS_LS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const KAPPAS = Object.keys(CENSUS.scales);
const LOCK_DELTAS = Object.keys(REL.lock);
const POS_MAX = CLOCKS.kernel[Object.keys(CLOCKS.kernel)[0]].length - 1;

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const baseLabel = (b: number) => `10${{ 100: '²', 10000: '⁴', 1000000: '⁶' }[b] ?? ''}`;
const fmt = (v: number | null | undefined, nd = 3) => (v === null || v === undefined ? 'n/a' : v.toFixed(nd));

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

function Readouts({ children }: { children: ReactNode }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>{children}</div>;
}

function Note({ children }: { children: ReactNode }) {
  return <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>{children}</p>;
}

function Legend({ items }: { items: { label: string; color: string; dash?: string }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="8" aria-hidden="true"><line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} /></svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

function Slider({ id, label, value, max, onChange, text }: { id: string; label: string; value: number; max: number; onChange: (v: number) => void; text: string }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', marginBottom: '0.45rem' }}>
      <label htmlFor={`positional-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`positional-lab-${id}`} type="range" min={0} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

/** One plot frame: gridlines and labels are drawn by the caller inside the translated group. */
const PW = 600, PH = 250, PM = { top: 22, right: 18, bottom: 34, left: 46 };
const IW = PW - PM.left - PM.right, IH = PH - PM.top - PM.bottom;

/** The latest ORG mention at or before each position, and the distance the query (one past it) reaches back. */
function lastMentions(labels: number[]): { last: number[]; gap: number[] } {
  let cur = -1;
  const last: number[] = [], gap: number[] = [];
  labels.forEach((lab, i) => { if (lab >= 0) cur = i; last.push(cur); gap.push(cur >= 0 ? i + 1 - cur : -1); });
  return { last, gap };
}

export default function PositionalEncodingLab() {
  const [viewA, setViewA] = useState<'clocks' | 'ruler' | 'relativity'>('clocks');
  const [baseIdx, setBaseIdx] = useState(1);
  const [dim, setDim] = useState(CLOCKS.dims[0]);
  const [m, setM] = useState(3);
  const [n, setN] = useState(10);
  const [lock, setLock] = useState(false);
  const [deltaIdx, setDeltaIdx] = useState(d3.bisectRight(RULER.deltas, 64) - 1); // the grid point at or below 64
  const [lockDelta, setLockDelta] = useState(LOCK_DELTAS[1]);
  const [viewB, setViewB] = useState<'tournament' | 'census' | 'filing'>('tournament');
  const [scheme, setScheme] = useState<Scheme>('rope');
  const [lIdx, setLIdx] = useState(CENSUS_LS.indexOf(CENSUS.L));
  const [rescue, setRescue] = useState<Rescue | 'none'>('none');
  const [kappa, setKappa] = useState(KAPPAS[1]);
  const [reader, setReader] = useState('alibi');
  const [pos, setPos] = useState(() => {
    const { gap } = lastMentions(FILING.labels);
    // the page's example: the first "the company" that every reader but RoPE + PI resolves wrongly
    const others = Object.keys(FILING.answers).filter((r) => r !== 'rope+pi');
    const missed = (t: number) => gap[t] > 1 && FILING.answers['rope+pi'][t] === FILING.truth[t] && others.every((r) => FILING.answers[r][t] !== FILING.truth[t]);
    const anaphor = FILING.tokens.findIndex((w, t) => w === 'company' && FILING.tokens[t - 1] === 'the' && missed(t));
    const i = anaphor >= 0 ? anaphor : FILING.truth.findIndex((_, t) => missed(t));
    return i >= 0 ? i : FILING.tokens.length - 1;
  });

  // ----- Figure A -----
  const base = CLOCKS.bases[baseIdx];
  const kern = CLOCKS.kernel[`${base}-${dim}`];
  const delta = n - m;
  const freqs = d3.range(dim / 2).map((j) => base ** (-2 * j / dim));
  const moveM = (v: number) => {
    if (!lock) return setM(v);
    if (v + delta >= 0 && v + delta <= POS_MAX) { setM(v); setN(v + delta); }
  };
  const moveN = (v: number) => {
    if (!lock) return setN(v);
    if (v - delta >= 0 && v - delta <= POS_MAX) { setN(v); setM(v - delta); }
  };
  const cols = 8, cell = PW / cols, rad = cell * 0.36, rows = Math.ceil(freqs.length / cols);
  const dialH = rows * (cell * 0.95) + 8;

  const rx = d3.scaleLog().domain([1, RULER.deltas[RULER.deltas.length - 1]]).range([0, IW]);
  const rVals = [...Object.values(RULER.kernel).flat(), ...RULER.limit, ...RULER.log_ruler];
  const ry = d3.scaleLinear().domain([Math.min(0, d3.min(rVals)!), 1]).range([IH, 0]).nice();
  const rLine = (arr: number[]) => d3.line<number>().x((i) => rx(RULER.deltas[i])).y((i) => ry(arr[i]))(d3.range(RULER.deltas.length))!;
  const RULER_SERIES = [
    { key: '16', label: 'd = 16', color: 'var(--color-text-secondary)', dash: '2 3', arr: RULER.kernel['16'] },
    { key: '64', label: 'd = 64', color: 'var(--color-badge-blue-text)', dash: '6 3', arr: RULER.kernel['64'] },
    { key: '512', label: 'd = 512', color: 'var(--color-accent-secondary)', dash: undefined, arr: RULER.kernel['512'] },
    { key: 'limit', label: 'the Ci limit (d → ∞)', color: 'var(--color-accent)', dash: undefined, arr: RULER.limit },
    { key: 'ruler', label: '1 − (γ + ln Δ)/ln b', color: 'var(--color-text)', dash: '8 3', arr: RULER.log_ruler },
  ];
  const dGrid = RULER.deltas[deltaIdx];

  const lk = REL.lock[lockDelta];
  const lx = d3.scaleLinear().domain([0, POS_MAX]).range([0, IW]);
  const lVals = [...lk.additive, lk.projected];
  const ly = d3.scaleLinear().domain([d3.min(lVals)!, d3.max(lVals)!]).range([IH, 0]).nice();

  // ----- Figure B -----
  const tx = d3.scaleLog().base(2).domain([TOUR.lengths[0], TOUR.lengths[TOUR.lengths.length - 1]]).range([0, IW]);
  const ty = d3.scaleLinear().domain([0, 1]).range([IH, 0]);
  const accLines = [
    ...SCHEMES.map((s) => ({ key: s, vals: TOUR.accuracy[s] })),
    ...RESCUES.map((r) => ({ key: `rope+${r}`, vals: TOUR.rescues[r] as (number | null)[] })),
  ];
  const curve = TOUR.offset_score[scheme];
  const ox = d3.scaleLinear().domain([0, TOUR.offset_score.rope.length - 1]).range([0, IW]);
  const oy = d3.scaleLinear().domain(d3.extent(curve) as [number, number]).range([IH, 0]).nice();
  const past = curve.slice(CENSUS.L);
  const best = past.length ? CENSUS.L + past.indexOf(Math.max(...past)) : null;
  const peakR = best !== null && curve[best] > curve[CENSUS.L - 1] ? best : null; // a rebound past the window, if any

  const L = CENSUS_LS[lIdx];
  const complete = CENSUS.wavelengths.map((lam) => lam <= L - 1);
  const cx = d3.scaleBand<number>().domain(d3.range(CENSUS.wavelengths.length)).range([0, IW]).padding(0.25);
  const cy = d3.scaleLog().domain([4, 40000]).range([IH, 0]);
  const scales = rescue === 'none' ? CENSUS.wavelengths.map(() => 1) : CENSUS.scales[kappa][rescue];
  const sy = d3.scaleLinear().domain([0, 1]).range([110 - 40, 0]);

  const { last, gap } = useMemo(() => lastMentions(FILING.labels), []);
  const names = useMemo(() => {
    const out: string[] = [];
    FILING.labels.forEach((lab, i) => { if (lab >= 0 && out[lab] === undefined) out[lab] = FILING.tokens[i]; });
    return out;
  }, []);
  const company = (lab: number) => (lab >= 0 ? `${LETTERS[lab]} · ${names[lab] ?? ''}` : 'none yet');
  const t = Math.min(pos, FILING.tokens.length - 1);
  const bx = d3.scalePoint<number>().domain(d3.range(FILING.bands.length)).range([0, IW]).padding(0.4);
  const bandLabel = (b: (number | null)[]) => (b[1] === null ? `${b[0]}+` : `${b[0]}–${b[1]}`);
  const horizonBand = FILING.bands.findIndex((b) => FILING.horizon >= (b[0] as number) && (b[1] === null || FILING.horizon <= (b[1] as number)));

  return (
    <div data-lab="positional" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Clocks and kernels — the sinusoidal code, the kernel it induces, and what a learned score does with it
        </figcaption>
        <Row>
          <Pill active={viewA === 'clocks'} onClick={() => setViewA('clocks')} data={{ 'data-view': 'clocks' }}>the clocks</Pill>
          <Pill active={viewA === 'ruler'} onClick={() => setViewA('ruler')} data={{ 'data-view': 'ruler' }}>the logarithmic ruler</Pill>
          <Pill active={viewA === 'relativity'} onClick={() => setViewA('relativity')} data={{ 'data-view': 'relativity' }}>is the learned score relative?</Pill>
        </Row>

        {viewA === 'clocks' && (
          <>
            <Row>
              {CLOCKS.bases.map((b, i) => (
                <Pill key={b} active={i === baseIdx} onClick={() => setBaseIdx(i)} data={{ 'data-base': String(b) }}>{`base b = ${baseLabel(b)}`}</Pill>
              ))}
              {CLOCKS.dims.map((d) => (
                <Pill key={d} active={d === dim} onClick={() => setDim(d)} data={{ 'data-dim': String(d) }}>{`d = ${d}`}</Pill>
              ))}
              <Pill active={lock} onClick={() => setLock(!lock)} data={{ 'data-control': 'lock' }}>{lock ? 'offset locked' : 'lock the offset'}</Pill>
            </Row>
            <Slider id="m" label="position m" value={m} max={POS_MAX} onChange={moveM} text={`m = ${m}`} />
            <Slider id="n" label="position n" value={n} max={POS_MAX} onChange={moveN} text={`n = ${n}`} />
            <svg viewBox={`0 0 ${PW} ${dialH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="clocks"
              aria-label={`${freqs.length} clock dials, one per plane, with hands at angle m times the plane's frequency (m = ${m}) and n times it (n = ${n})`}>
              {freqs.map((w, j) => {
                const cxj = cell * (j % cols) + cell / 2, cyj = cell * 0.95 * Math.floor(j / cols) + cell * 0.47;
                const hand = (p: number) => [cxj + rad * Math.cos(p * w), cyj - rad * Math.sin(p * w)];
                const [mx, my] = hand(m), [nx, ny] = hand(n);
                return (
                  <g key={j} data-dial={j}>
                    <circle cx={cxj} cy={cyj} r={rad} style={{ fill: 'none', stroke: 'var(--color-border)', strokeWidth: 1 }} />
                    <line x1={cxj} y1={cyj} x2={mx} y2={my} style={{ stroke: 'var(--color-accent)', strokeWidth: 2.2 }} />
                    <line x1={cxj} y1={cyj} x2={nx} y2={ny} style={{ stroke: 'var(--color-accent-secondary)', strokeWidth: 2.2, strokeDasharray: '3 2' }} />
                    <text x={cxj} y={cyj + rad + 11} textAnchor="middle" style={axisText}>{`plane ${j}`}</text>
                  </g>
                );
              })}
            </svg>
            <Legend items={[{ label: 'hand at m', color: 'var(--color-accent)' }, { label: 'hand at n', color: 'var(--color-accent-secondary)', dash: '3 2' }]} />
            <Readouts>
              <Readout id="offset" label="offset n − m" value={String(delta)} />
              <Readout id="kernel" label="⟨p(m), p(n)⟩ / (d/2)" value={kern[Math.abs(delta)].toFixed(3)} emphasis />
            </Readouts>
            <Note>
              Plane j turns at rate b^(−2j/d): the first hand at one radian per step, the last almost not at all. Lock the
              offset and move either slider: every hand turns, each by its own rate, and the readout does not move. The
              inner product of two positions depends only on how far apart they are (Proposition 2 and Theorem 1).
            </Note>
          </>
        )}

        {viewA === 'ruler' && (
          <>
            <Slider id="delta" label="distance Δ" value={deltaIdx} max={RULER.deltas.length - 1} onChange={setDeltaIdx} text={`Δ = ${dGrid}`} />
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="ruler"
              aria-label="The normalized sinusoidal kernel against distance on a log scale, for d = 16, 64, and 512, beside its cosine-integral limit and the logarithmic ruler">
              <g transform={`translate(${PM.left},${PM.top})`}>
                {ry.ticks(5).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={ry(v)} y2={ry(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={ry(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {[1, 4, 16, 64, 256, 1024, 4096].map((v) => (
                  <text key={v} x={rx(v)} y={IH + 14} textAnchor="middle" style={axisText}>{v}</text>
                ))}
                {RULER_SERIES.map((s) => (
                  <path key={s.key} d={rLine(s.arr)} data-series={s.key}
                    style={{ fill: 'none', stroke: s.color, strokeWidth: s.key === 'limit' ? 2.5 : 1.6, strokeDasharray: s.dash }} />
                ))}
                <line x1={rx(dGrid)} x2={rx(dGrid)} y1={0} y2={IH} style={{ stroke: 'var(--color-text)', strokeWidth: 1, strokeDasharray: '3 3' }} />
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>⟨p(m), p(m + Δ)⟩ / (d/2), base b = 10⁴</text>
              <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>distance Δ (log scale)</text>
            </svg>
            <Legend items={RULER_SERIES.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />
            <Readouts>
              {RULER_SERIES.map((s) => (
                <Readout key={s.key} id={`ruler-${s.key}`} label={s.label} value={s.arr[deltaIdx].toFixed(3)} emphasis={s.key === 'limit'} />
              ))}
            </Readouts>
            <Note>
              As d grows the kernel settles onto the cosine-integral limit of Theorem 1 (ii), and past Δ ≈ 16 that limit is a
              straight line in log Δ: each doubling of distance costs ln 2/ln b ≈ 0.075 of similarity. At d = 64 and large Δ
              the finite sum still wanders off the limit.
            </Note>
          </>
        )}

        {viewA === 'relativity' && (
          <>
            <Row>
              {LOCK_DELTAS.map((k) => (
                <Pill key={k} active={k === lockDelta} onClick={() => setLockDelta(k)} data={{ 'data-lock': k }}>{`offset Δ = ${k}`}</Pill>
              ))}
            </Row>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="relativity"
              aria-label={`The trained sinusoidal head's position-to-position score at offset ${lockDelta} as the position m slides, against its commutant projection`}>
              <g transform={`translate(${PM.left},${PM.top})`}>
                {ly.ticks(5).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={ly(v)} y2={ly(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={ly(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {[0, 16, 32, 48, 63].map((v) => <text key={v} x={lx(v)} y={IH + 14} textAnchor="middle" style={axisText}>{v}</text>)}
                <path d={d3.line<number>().x((_, i) => lx(i)).y((v) => ly(v))(lk.additive)!} data-series="additive"
                  style={{ fill: 'none', stroke: 'var(--color-badge-blue-text)', strokeWidth: 2 }} />
                <line x1={0} x2={IW} y1={ly(lk.projected)} y2={ly(lk.projected)} data-series="projected"
                  style={{ stroke: 'var(--color-accent)', strokeWidth: 2, strokeDasharray: '6 3' }} />
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>{`p(m)ᵀ M p(m + ${lockDelta}) / τ, the trained head's M = W_Q W_Kᵀ`}</text>
              <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>position m (the offset held fixed)</text>
            </svg>
            <Legend items={[{ label: 'the trained M', color: 'var(--color-badge-blue-text)' }, { label: 'its commutant projection Π(M)', color: 'var(--color-accent)', dash: '6 3' }]} />
            <Readouts>
              <Readout id="defect" label="relativity defect δ(M)" value={REL.defect.toFixed(3)} emphasis />
              <Readout id="additive-range" label="trained M, as m slides" value={`${Math.min(...lk.additive).toFixed(3)} to ${Math.max(...lk.additive).toFixed(3)}`} />
              <Readout id="projected" label="Π(M), at every m" value={lk.projected.toFixed(3)} />
            </Readouts>
            <Note>
              A score that sees only the offset would be flat here. Theorem 2 says only the d-dimensional commutant of the
              rotations gives one; the trained head's M sits as far from it as a random matrix (a random 16 × 16 matrix has
              δ ≈ 0.968), so its score moves with absolute position, and the head fails as soon as positions pass the window.
            </Note>
          </>
        )}
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Past the training window — heads trained at 32 tokens, read longer
        </figcaption>
        <Row>
          <Pill active={viewB === 'tournament'} onClick={() => setViewB('tournament')} data={{ 'data-view': 'tournament' }}>the tournament</Pill>
          <Pill active={viewB === 'census'} onClick={() => setViewB('census')} data={{ 'data-view': 'census' }}>the wavelength census</Pill>
          <Pill active={viewB === 'filing'} onClick={() => setViewB('filing')} data={{ 'data-view': 'filing' }}>a filing</Pill>
        </Row>

        {viewB === 'tournament' && (
          <>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="accuracy"
              aria-label="Accuracy on the cue task against test length for every scheme and every rescaling of the RoPE head, mean of three seeds">
              <g transform={`translate(${PM.left},${PM.top})`}>
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={ty(v)} y2={ty(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={ty(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {TOUR.lengths.map((v) => <text key={v} x={tx(v)} y={IH + 14} textAnchor="middle" style={axisText}>{v}</text>)}
                {accLines.map(({ key, vals }) => {
                  const pts = vals.map((v, i) => (v === null ? null : [tx(TOUR.lengths[i]), ty(v)] as [number, number])).filter((p): p is [number, number] => p !== null);
                  const st = STYLE[key];
                  return (
                    <g key={key} data-series={key} style={{ opacity: key === scheme || key.startsWith('rope+') ? 1 : 0.75 }}>
                      {pts.length > 1 && <path d={d3.line()(pts)!} style={{ fill: 'none', stroke: st.color, strokeWidth: key === scheme ? 3 : 1.8, strokeDasharray: st.dash }} />}
                      {pts.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={key.startsWith('rope+') ? 2.5 : 3.5} style={{ fill: st.color }} />)}
                    </g>
                  );
                })}
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>accuracy on the cue task (mean of 3 seeds)</text>
              <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>test length (log scale) · trained at 32</text>
            </svg>
            <Legend items={accLines.map(({ key }) => ({ label: STYLE[key].label, color: STYLE[key].color, dash: STYLE[key].dash }))} />
            <Row style={{ marginTop: '0.6rem' }}>
              {SCHEMES.map((s) => (
                <Pill key={s} active={s === scheme} onClick={() => setScheme(s)} data={{ 'data-scheme': s }}>{STYLE[s].label}</Pill>
              ))}
            </Row>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="offset-score"
              aria-label={`The ${STYLE[scheme].label} head's score for a cue at offset r, with the offsets seen in training shaded`}>
              <g transform={`translate(${PM.left},${PM.top})`}>
                <rect x={0} y={0} width={ox(CENSUS.L)} height={IH} style={{ fill: 'var(--color-border)', opacity: 0.45 }} />
                <text x={ox(CENSUS.L / 2)} y={12} textAnchor="middle" style={axisText}>seen in training</text>
                {oy.ticks(5).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={oy(v)} y2={oy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={oy(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {[0, 32, 64, 128, 192, 255].map((v) => <text key={v} x={ox(v)} y={IH + 14} textAnchor="middle" style={axisText}>{v}</text>)}
                <path d={d3.line<number>().x((_, i) => ox(i)).y((v) => oy(v))(curve)!} data-series="offset-score"
                  style={{ fill: 'none', stroke: STYLE[scheme].color, strokeWidth: 2 }} />
                {peakR !== null && <circle cx={ox(peakR)} cy={oy(curve[peakR])} r={5} style={{ fill: 'none', stroke: 'var(--color-text)', strokeWidth: 1.5 }} />}
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>{`${STYLE[scheme].label}: score for a cue at offset r (seed 0)`}</text>
              <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>offset r back from the query</text>
            </svg>
            <Readouts>
              {TOUR.lengths.map((len, i) => (
                <Readout key={len} id={`acc-${len}`} label={`${STYLE[scheme].label} at ${len}`} value={fmt(TOUR.accuracy[scheme][i])} emphasis={len === TOUR.lengths[TOUR.lengths.length - 1]} />
              ))}
            </Readouts>
            <Readouts>
              <Readout id="score-1" label="score at r = 1" value={curve[1].toFixed(2)} />
              <Readout id="score-edge" label={`score at r = ${CENSUS.L - 1}, the window's edge`} value={curve[CENSUS.L - 1].toFixed(2)} />
              {peakR !== null && <Readout id="peak" label="it rises again: highest score past the window" value={`${curve[peakR].toFixed(2)} at r = ${peakR}`} emphasis />}
            </Readouts>
            <Note>
              Training only sees offsets 0 to 31. The RoPE head's score is a trigonometric polynomial in r (Proposition 3): it falls
              across the window, as the task rewards, and past it the planes that never completed a turn bring it back up, until a
              cue 181 positions back outscores a recent one. ALiBi's score is a straight line down. The absolute codes'
              curves are drawn for a query at the end of a 256-token sequence; the learned table has nothing past 31.
            </Note>
          </>
        )}

        {viewB === 'census' && (
          <>
            <Slider id="train-length" label="training length L" value={lIdx} max={CENSUS_LS.length - 1} onChange={setLIdx} text={`L = ${L}`} />
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="census"
              aria-label={`The wavelength of each RoPE plane at d = 16, against the training length ${L}: planes whose wavelength fits complete a turn`}>
              <g transform={`translate(${PM.left},${PM.top})`}>
                {[10, 100, 1000, 10000].map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={cy(v)} y2={cy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={cy(v) + 3} textAnchor="end" style={axisText}>{v.toLocaleString('en-US')}</text>
                  </g>
                ))}
                {CENSUS.wavelengths.map((lam, j) => (
                  <g key={j} data-plane={j} data-complete={String(complete[j])}>
                    <rect x={cx(j)} y={cy(lam)} width={cx.bandwidth()} height={IH - cy(lam)}
                      style={{ fill: complete[j] ? 'var(--color-accent)' : 'var(--color-accent-secondary)', opacity: complete[j] ? 0.9 : 0.45 }} />
                    <text x={cx(j)! + cx.bandwidth() / 2} y={IH + 14} textAnchor="middle" style={axisText}>{`plane ${j}`}</text>
                  </g>
                ))}
                <line x1={0} x2={IW} y1={cy(Math.max(L - 1, 4))} y2={cy(Math.max(L - 1, 4))} style={{ stroke: 'var(--color-text)', strokeWidth: 1.5, strokeDasharray: '5 3' }} />
                <text x={IW} y={cy(Math.max(L - 1, 4)) - 4} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>{`L − 1 = ${L - 1}`}</text>
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>wavelength λⱼ = 2π/θⱼ of each plane (log scale), d = 16, b = 10⁴</text>
            </svg>
            <Readouts>
              <Readout id="complete" label={`planes that turn a whole circle within ${L} positions`} value={`${complete.filter(Boolean).length} of ${complete.length}`} emphasis />
            </Readouts>
            <Row style={{ marginTop: '0.6rem' }}>
              {(['none', ...RESCUES] as const).map((r) => (
                <Pill key={r} active={r === rescue} onClick={() => setRescue(r)} data={{ 'data-rescue': r }}>{RESCUE_LABEL[r]}</Pill>
              ))}
              {KAPPAS.map((k) => (
                <Pill key={k} active={k === kappa} onClick={() => setKappa(k)} data={{ 'data-kappa': k }}>{`κ = ${k}`}</Pill>
              ))}
            </Row>
            <svg viewBox={`0 0 ${PW} 110`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="scales"
              aria-label={`The factor each plane's frequency is multiplied by under ${RESCUE_LABEL[rescue]} at kappa = ${kappa}`}>
              <g transform={`translate(${PM.left},20)`}>
                {scales.map((v, j) => (
                  <g key={j}>
                    <rect x={cx(j)} y={sy(v)} width={cx.bandwidth()} height={sy(0) - sy(v)} style={{ fill: 'var(--color-accent)', opacity: 0.8 }} />
                    <text x={cx(j)! + cx.bandwidth() / 2} y={sy(v) - 3} textAnchor="middle" style={axisText}>{v.toFixed(3)}</text>
                  </g>
                ))}
                <line x1={0} x2={IW} y1={sy(0)} y2={sy(0)} style={{ stroke: 'var(--color-border)' }} />
              </g>
              <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>{`frequency scale per plane, ${RESCUE_LABEL[rescue]}${rescue === 'none' ? '' : `, κ = ${kappa}`}`}</text>
            </svg>
            <Note>
              A plane whose wavelength exceeds L − 1 never turns a whole circle in training, so past L it shows the head angles it
              has never seen (Theorem 5 (i)). PI divides every frequency by κ, which keeps all angles in the trained range and
              blurs neighbors κ-fold; NTK-aware scaling leaves plane 0 alone and gives the slowest plane exactly 1/κ; YaRN
              interpolates planes by how many turns they made in training.
            </Note>
          </>
        )}

        {viewB === 'filing' && (
          <>
            <Row>
              {READERS.map((r) => (
                <Pill key={r} active={r === reader} onClick={() => setReader(r)} data={{ 'data-reader': r }}>{STYLE[r].label}</Pill>
              ))}
            </Row>
            <div data-panel="filing" style={{ fontFamily: sans, fontSize: '0.8rem', lineHeight: 1.9, maxHeight: '15rem', overflowY: 'auto',
              border: '1px solid var(--color-border)', borderRadius: '0.5rem', padding: '0.4rem 0.5rem', overflowWrap: 'anywhere' }}>
              {FILING.tokens.map((w, i) => {
                const lab = FILING.labels[i];
                const wrong = FILING.truth[i] >= 0 && FILING.answers[reader][i] !== FILING.truth[i];
                return (
                  <span key={i} data-token={i} onClick={() => setPos(i)}
                    title={FILING.truth[i] >= 0 ? `asked here: ${company(FILING.truth[i])}, ${gap[i]} back; ${STYLE[reader].label} says ${company(FILING.answers[reader][i])}` : w}
                    style={{
                      display: 'inline-block', margin: '0 0.12rem', padding: '0 0.22rem', borderRadius: '0.25rem', cursor: 'pointer',
                      maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', verticalAlign: 'bottom',
                      color: lab >= 0 ? 'var(--color-bg)' : 'var(--color-text)',
                      background: lab >= 0 ? COMPANY_COLORS[lab % COMPANY_COLORS.length] : 'transparent',
                      outline: i === t ? '2px solid var(--color-text)' : i === last[t] ? '2px dashed var(--color-accent)' : 'none',
                      borderBottom: wrong ? '3px solid var(--color-badge-red-text)' : '3px solid transparent',
                    }}>
                    {w}
                  </span>
                );
              })}
            </div>
            <Note>
              {`Held-out filing ${FILING.index}. Colored chips: company mentions, by company. Click any word to ask “which company was named last?” just after it. A red underline marks the positions the chosen reader gets wrong (heads trained on 32-token windows of the training filings, seed 0).`}
            </Note>
            <Readouts>
              <Readout id="truth" label="named last" value={company(FILING.truth[t])} emphasis />
              <Readout id="gap" label="how far back" value={gap[t] >= 0 ? `${gap[t]} tokens` : '—'} />
            </Readouts>
            <div data-panel="answers" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1rem', fontFamily: sans, fontSize: '0.78rem', marginTop: '0.5rem' }}>
              {READERS.map((r) => {
                const a = FILING.answers[r][t];
                const ok = FILING.truth[t] < 0 || a === FILING.truth[t];
                return (
                  <span key={r} data-answer={r} style={{ minWidth: 0, color: ok ? 'var(--color-text)' : 'var(--color-badge-red-text)' }}>
                    {`${STYLE[r].label}: ${FILING.truth[t] < 0 ? '—' : `${LETTERS[a] ?? '?'} ${ok ? '✓' : '✗'}`}`}
                  </span>
                );
              })}
            </div>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto', marginTop: '0.6rem' }} role="img" data-panel="bands"
              aria-label="Accuracy on all 100 held-out filings by how far back the answer is, for every reader, mean of three seeds">
              <g transform={`translate(${PM.left},${PM.top})`}>
                {[0, 0.25, 0.5, 0.75, 1].map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={ty(v)} y2={ty(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={ty(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {FILING.bands.map((b, i) => <text key={i} x={bx(i)} y={IH + 14} textAnchor="middle" style={axisText}>{bandLabel(b)}</text>)}
                {horizonBand >= 0 && (
                  <g data-series="horizon">
                    <line x1={bx(horizonBand)} x2={bx(horizonBand)} y1={0} y2={IH} style={{ stroke: 'var(--color-accent-secondary)', strokeWidth: 1.2, strokeDasharray: '4 3' }} />
                    <text x={bx(horizonBand)! + 4} y={IH - 6} style={{ ...axisText, fill: 'var(--color-accent-secondary)' }}>{`ALiBi’s horizon ≈ ${FILING.horizon}`}</text>
                  </g>
                )}
                {BAND_SERIES.map((r) => (
                  <g key={r} data-series={r}>
                    <path d={d3.line<number>().x((_, i) => bx(i)!).y((v) => ty(v))(FILING.band_accuracy[r])!}
                      style={{ fill: 'none', stroke: STYLE[r].color, strokeWidth: r === reader ? 3 : 1.6, strokeDasharray: STYLE[r].dash }} />
                    {FILING.band_accuracy[r].map((v, i) => <circle key={i} cx={bx(i)} cy={ty(v)} r={r === reader ? 3.5 : 2.5} style={{ fill: STYLE[r].color }} />)}
                  </g>
                ))}
              </g>
              <text x={8} y={13} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>accuracy on all held-out filings, by how far back the answer is</text>
              <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>distance to the last company mention (tokens) · trained on windows of 32</text>
            </svg>
            <Legend items={BAND_SERIES.map((r) => ({ label: STYLE[r].label, color: STYLE[r].color, dash: STYLE[r].dash }))} />
            <Readouts>
              {FILING.bands.map((b, i) => (
                <Readout key={i} id={`band-${i}`} label={`${STYLE[reader].label}, ${bandLabel(b)} back`} value={FILING.band_accuracy[reader][i].toFixed(3)} emphasis={i >= 3} />
              ))}
            </Readouts>
            <Note>
              Inside the training window RoPE and ALiBi are nearly perfect, and NoPE and the sinusoidal code are not. Past it,
              ALiBi falls off at its horizon (Theorem 3 (iii): a mention farther back than W/s weighs less than the query’s own
              position, whatever it says), plain RoPE and NTK-aware scaling meet angles the head never saw, and RoPE with
              interpolated positions (PI, YaRN) reads the whole filing.
            </Note>
          </>
        )}
      </figure>
    </div>
  );
}
