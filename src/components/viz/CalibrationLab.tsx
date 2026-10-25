import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/language-model-calibration/language_model_calibration.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_binning_hides, test_calibrated_twin_floor, test_temperature_keeps_argmax_not_ranking,
// test_kneser_ney_reliability_shape, test_in_distribution_nearly_calibrated, test_shift_can_go_either_way,
// test_edgar_overconfidence, test_fact_regimes_pull_apart). Change the .py, regenerate, never the reverse.
import payload from './data/language-model-calibration-lab.json';

/**
 * Calibration Laboratory — two figures for the language-model-calibration topic.
 *   A. "Reliability": either corpus engine on any of three test sets, tempered at any of eleven τ, binned at any of six
 *      bin counts. Left, the reliability diagram (Definition 2); right, the plug-in ECE against the bin count beside
 *      the calibrated twin's noise band (Theorem 2). Readouts: NLL, the nucleus(0.9) coverage against its kept mass
 *      (Proposition 3), τ*, and the cross-fitted temperature gain (Corollary 1).
 *   B. "Facts by regime": Topic 15's order-5 fact model on 111 probes split by training count; confidence against
 *      accuracy and the log loss of the knowledge base's answer as τ slides.
 */

type Engine = 'trigram' | 'decoder';
type TestSet = 'continuation' | 'held' | 'edgar';
type Point = {
  nll: number; acc: number; conf: number; cover: number; kept: number;
  ece: number[]; twin: number[]; twin_sd: number[]; bins: Record<string, [number, number, number][]>;
};
type SetData = { n: number; tau_star: number; nll_star: number; gain: number; points: Point[] };
const TAUS = payload.taus as number[];
const BINS = payload.bins as number[];
const ENGINES = payload.engines as unknown as Record<Engine, { V: number; sets: Record<TestSet, SetData> }>;
type Regime = 'seen' | 'once' | 'unseen';
const REGIMES: Regime[] = ['seen', 'once', 'unseen'];
type FactCurve = { n: number; acc: number; tau_star: number; conf: number[]; p_correct: number[]; nll: number[] };
const FACTS = payload.facts as unknown as Record<Regime | 'all', FactCurve> & {
  taus: number[]; count_aware: { tau1: number; pooled: number; per_regime: number };
};
const FTAUS = FACTS.taus;
const ONE = TAUS.indexOf(1);
const FONE = FTAUS.indexOf(1);

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3);
const tauText = (t: number) => (t >= 1 ? t.toFixed(2) : t.toFixed(3));

const ENGINE_LABEL: Record<Engine, string> = { trigram: 'KN trigram', decoder: 'decoder' };
const SET_LABEL: Record<TestSet, string> = {
  continuation: 'continuation (same generator)', held: 'held-out split', edgar: 'EDGAR filings',
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

function Legend({ items }: { items: { label: string; color: string; dash?: string; band?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="10" aria-hidden="true">
            {s.band
              ? <rect x="0" y="1" width="22" height="8" style={{ fill: s.color, opacity: 0.25 }} />
              : <line x1="0" y1="5" x2="22" y2="5" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />}
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
      <label htmlFor={`cal-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`cal-lab-${id}`} type="range" min={0} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

// ----- Figure A, left: the reliability diagram -----

const RW = 600, RH = 360, RM = { top: 24, right: 14, bottom: 36, left: 44 };
const RIW = RW - RM.left - RM.right, RIH = RH - RM.top - RM.bottom;

function Reliability({ bins, n, aria }: { bins: [number, number, number][]; n: number; aria: string }) {
  const x = d3.scaleLinear().domain([0, 1]).range([0, RIW]);
  const y = d3.scaleLinear().domain([0, 1]).range([RIH, 0]);
  const r = d3.scaleSqrt().domain([0, n]).range([0, 26]);
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  return (
    <svg viewBox={`0 0 ${RW} ${RH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="reliability" aria-label={aria}>
      <text x={8} y={14} style={titleText}>accuracy against confidence, one circle per bin (area ∝ tokens); ticks mark each gap</text>
      <g transform={`translate(${RM.left},${RM.top})`}>
        {ticks.map((t) => (
          <g key={`g${t}`}>
            <line x1={0} x2={RIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t}</text>
            <text x={x(t)} y={RIH + 14} textAnchor="middle" style={axisText}>{t}</text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '4 3' }} />
        {bins.map(([k, c, a], i) => (
          <g key={`b${i}`} data-bin={i} data-count={k}>
            <line x1={x(c)} x2={x(c)} y1={y(c)} y2={y(a)} style={{ stroke: 'var(--color-badge-red-text)', strokeWidth: 1.5 }} />
            <circle cx={x(c)} cy={y(a)} r={Math.max(2, r(k))} style={{ fill: 'var(--color-accent)', fillOpacity: 0.35, stroke: 'var(--color-accent)', strokeWidth: 1.2 }} />
          </g>
        ))}
      </g>
      <text x={RM.left + RIW / 2} y={RH - 4} textAnchor="middle" style={axisText}>mean confidence in the bin</text>
      <text transform={`translate(11,${RM.top + RIH / 2}) rotate(-90)`} textAnchor="middle" style={axisText}>accuracy</text>
    </svg>
  );
}

// ----- Figure A, right: ECE against the bin count, beside the calibrated twin -----

const EW = 600, EH = 220, EM = { top: 24, right: 14, bottom: 36, left: 44 };
const EIW = EW - EM.left - EM.right, EIH = EH - EM.top - EM.bottom;

function EceCurve({ P, bi }: { P: Point; bi: number }) {
  const x = d3.scaleLog().domain([1, BINS[BINS.length - 1]]).range([0, EIW]);
  const top = Math.max(...P.ece, ...P.twin.map((m, i) => m + 2 * P.twin_sd[i]));
  const y = d3.scaleLinear().domain([0, top * 1.1]).range([EIH, 0]).nice();
  const line = d3.line<number>().x((_, i) => x(BINS[i])).y((v) => y(v));
  const band = d3.area<number>().x((_, i) => x(BINS[i]))
    .y0((m, i) => y(Math.max(0, m - 2 * P.twin_sd[i]))).y1((m, i) => y(m + 2 * P.twin_sd[i]));
  return (
    <svg viewBox={`0 0 ${EW} ${EH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="ece-curve"
      aria-label={`At ${BINS[bi]} bins: ECE ${f3(P.ece[bi])}, calibrated twin ${f3(P.twin[bi])}`}>
      <text x={8} y={14} style={titleText}>plug-in ECE against the bin count, and the calibrated twin’s mean ± 2 SD</text>
      <g transform={`translate(${EM.left},${EM.top})`}>
        {y.ticks(4).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={EIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{String(Number(t.toFixed(3)))}</text>
          </g>
        ))}
        {BINS.map((B) => (
          <text key={`x${B}`} x={x(B)} y={EIH + 14} textAnchor="middle" style={axisText}>{B}</text>
        ))}
        <line x1={x(BINS[bi])} x2={x(BINS[bi])} y1={0} y2={EIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '2 3' }} />
        <path d={band(P.twin) ?? ''} style={{ fill: 'var(--color-accent-secondary)', opacity: 0.25 }} />
        <path d={line(P.twin) ?? ''} data-series="twin" style={{ fill: 'none', stroke: 'var(--color-accent-secondary)', strokeWidth: 1.6, strokeDasharray: '5 3' }} />
        <path d={line(P.ece) ?? ''} data-series="ece" style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
        {P.ece.map((v, i) => (
          <circle key={`p${i}`} cx={x(BINS[i])} cy={y(v)} r={i === bi ? 4.5 : 3} style={{ fill: 'var(--color-accent)' }} />
        ))}
      </g>
      <text x={EM.left + EIW / 2} y={EH - 4} textAnchor="middle" style={axisText}>number of equal-width bins B (log scale)</text>
    </svg>
  );
}

// ----- Figure B: facts by regime -----

const FW = 600, FH = 250, FM = { top: 24, right: 14, bottom: 36, left: 44 };
const FIW = FW - FM.left - FM.right, FIH = FH - FM.top - FM.bottom;
const xF = d3.scaleLog().domain([FTAUS[0], FTAUS[FTAUS.length - 1]]).range([0, FIW]);

// Cold samplers make never-seen facts cost tens of nats; the chart clips at FACT_NLL_MAX so the regimes stay legible.
const FACT_NLL_MAX = 4;

function FactNll({ ti }: { ti: number }) {
  const y = d3.scaleLinear().domain([0, FACT_NLL_MAX]).range([FIH, 0]);
  const line = d3.line<number>().x((_, i) => xF(FTAUS[i])).y((v) => y(v));
  return (
    <svg viewBox={`0 0 ${FW} ${FH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="fact-nll"
      aria-label={`At τ = ${tauText(FTAUS[ti])}: ${REGIMES.map((g) => `${REGIME_LABEL[g]} log loss ${f3(FACTS[g].nll[ti])}`).join('; ')}`}>
      <defs>
        <clipPath id="cal-lab-fact-clip"><rect x={0} y={0} width={FIW} height={FIH} /></clipPath>
      </defs>
      <text x={8} y={14} style={titleText}>−log P(the knowledge base’s answer) per regime, clipped at {FACT_NLL_MAX} nats; triangles mark each best τ</text>
      <g transform={`translate(${FM.left},${FM.top})`}>
        {y.ticks(4).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={FIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{String(Number(t.toFixed(2)))}</text>
          </g>
        ))}
        {[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].map((t) => (
          <text key={`x${t}`} x={xF(t)} y={FIH + 14} textAnchor="middle" style={{ ...axisText, fontWeight: t === 1 ? 700 : 400 }}>{t}</text>
        ))}
        <line x1={xF(FTAUS[ti])} x2={xF(FTAUS[ti])} y1={0} y2={FIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '2 3' }} />
        <g clipPath="url(#cal-lab-fact-clip)">
          <path d={line(FACTS.all.nll) ?? ''} data-series="all" style={{ fill: 'none', stroke: 'var(--color-text-secondary)', strokeWidth: 1.6, strokeDasharray: '5 3' }} />
          {REGIMES.map((g) => (
            <path key={g} d={line(FACTS[g].nll) ?? ''} data-series={g} style={{ fill: 'none', stroke: REGIME_COLOR[g], strokeWidth: 2 }} />
          ))}
        </g>
        {REGIMES.map((g) => (
          <path key={`m${g}`} d={`M${xF(FACTS[g].tau_star)},${FIH} l-5,8 l10,0 z`} style={{ fill: REGIME_COLOR[g] }} />
        ))}
        <path d={`M${xF(FACTS.all.tau_star)},${FIH} l-5,8 l10,0 z`} style={{ fill: 'var(--color-text-secondary)' }} />
      </g>
      <text x={FM.left + FIW / 2} y={FH - 4} textAnchor="middle" style={axisText}>temperature τ (log scale)</text>
    </svg>
  );
}

const CW = 600, CH = 150, CM = { top: 24, right: 14, bottom: 30, left: 128 };
const CIW = CW - CM.left - CM.right, CIH = CH - CM.top - CM.bottom;

function FactConfidence({ ti }: { ti: number }) {
  const x = d3.scaleLinear().domain([0, 1]).range([0, CIW]);
  const y = d3.scaleBand<Regime>().domain(REGIMES).range([0, CIH]).padding(0.35);
  return (
    <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="fact-confidence"
      aria-label={`At τ = ${tauText(FTAUS[ti])}: ${REGIMES.map((g) => `${REGIME_LABEL[g]} confidence ${f3(FACTS[g].conf[ti])}, accuracy ${f3(FACTS[g].acc)}`).join('; ')}`}>
      <text x={8} y={14} style={titleText}>mean confidence (bar) against accuracy (black tick), by regime</text>
      <g transform={`translate(${CM.left},${CM.top})`}>
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={`x${t}`}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={CIH} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={x(t)} y={CIH + 14} textAnchor="middle" style={axisText}>{t}</text>
          </g>
        ))}
        {REGIMES.map((g) => (
          <g key={g} data-regime={g}>
            <text x={-8} y={y(g)! + y.bandwidth() / 2 + 3} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>{REGIME_LABEL[g]}</text>
            <rect x={0} y={y(g)} width={x(FACTS[g].conf[ti])} height={y.bandwidth()} style={{ fill: REGIME_COLOR[g], opacity: 0.8 }} />
            <line x1={x(FACTS[g].acc)} x2={x(FACTS[g].acc)} y1={y(g)! - 4} y2={y(g)! + y.bandwidth() + 4}
              style={{ stroke: 'var(--color-text)', strokeWidth: 2.5 }} />
          </g>
        ))}
      </g>
    </svg>
  );
}

export default function CalibrationLab() {
  const [engine, setEngine] = useState<Engine>('trigram');
  const [set, setSet] = useState<TestSet>('continuation');
  const [ti, setTi] = useState(ONE);
  const [bi, setBi] = useState(BINS.indexOf(15));
  const [fti, setFti] = useState(FONE);

  const S = ENGINES[engine].sets[set];
  const P = S.points[ti];
  const B = BINS[bi];
  const ca = FACTS.count_aware;

  return (
    <div data-lab="calibration" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <Caption>A · Reliability — how honest the top probability is, and how much of the measured error is noise</Caption>
        <Row>
          {(['trigram', 'decoder'] as Engine[]).map((e) => (
            <Pill key={e} active={engine === e} onClick={() => setEngine(e)} data={{ 'data-engine': e }}>
              {`${ENGINE_LABEL[e]} (V = ${ENGINES[e].V.toLocaleString('en-US')})`}
            </Pill>
          ))}
        </Row>
        <Row>
          {(['continuation', 'held', 'edgar'] as TestSet[]).map((t) => (
            <Pill key={t} active={set === t} onClick={() => setSet(t)} data={{ 'data-set': t }}>{SET_LABEL[t]}</Pill>
          ))}
        </Row>
        <Slider id="tau" label="temperature τ" value={ti} max={TAUS.length - 1} onChange={setTi} text={tauText(TAUS[ti])} />
        <Row style={{ alignItems: 'center' }}>
          <span style={{ fontFamily: sans, fontSize: '0.8rem' }}>bins B</span>
          {BINS.map((b, i) => (
            <Pill key={b} active={bi === i} onClick={() => setBi(i)} data={{ 'data-bins': String(b) }}>{String(b)}</Pill>
          ))}
        </Row>
        <Reliability bins={P.bins[String(B)]} n={S.n}
          aria={`${ENGINE_LABEL[engine]} on ${SET_LABEL[set]} at τ = ${tauText(TAUS[ti])}, ${B} bins: confidence ${f3(P.conf)}, accuracy ${f3(P.acc)}`} />
        <EceCurve P={P} bi={bi} />
        <Legend items={[
          { label: 'the engine’s ECE', color: 'var(--color-accent)' },
          { label: 'calibrated twin, mean', color: 'var(--color-accent-secondary)', dash: '5 3' },
          { label: `twin ± 2 SD (${payload.twin_r} twins)`, color: 'var(--color-accent-secondary)', band: true },
        ]} />
        <Readouts>
          <Readout id="ece" label={`ECE at ${B} bins`} value={f3(P.ece[bi])} emphasis />
          <Readout id="twin" label="calibrated twin’s ECE (noise floor)" value={`${f3(P.twin[bi])} ± ${f3(P.twin_sd[bi])}`} />
          <Readout id="conf-acc" label="mean confidence / accuracy" value={`${f3(P.conf)} / ${f3(P.acc)}`} />
          <Readout id="nll" label="log loss, nats per token" value={f3(P.nll)} />
          <Readout id="cover" label="nucleus(0.9): coverage / kept mass" value={`${f3(P.cover)} / ${f3(P.kept)}`} />
          <Readout id="tau-star" label="NLL-optimal τ* (its NLL)" value={`${f3(S.tau_star)} (${f3(S.nll_star)})`} />
          <Readout id="gain" label="cross-fitted gain from τ*, nats" value={f3(S.gain)} />
          <Readout id="n" label="tokens scored" value={S.n.toLocaleString('en-US')} />
        </Readouts>
        <Note>
          Fewer bins average miscalibration away and more bins invent it (Theorem 2): the twin, calibrated by construction,
          shows how much of an ECE is noise at this sample size. Temperature moves every circle sideways but never changes
          the accuracy. In distribution both engines sit near the diagonal; on the EDGAR filings no temperature brings the
          nucleus’s coverage near its kept mass (Proposition 3).
        </Note>
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <Caption>B · Facts by regime — one temperature cannot calibrate what the model has and has not seen</Caption>
        <Slider id="fact-tau" label="temperature τ" value={fti} max={FTAUS.length - 1} onChange={setFti} text={tauText(FTAUS[fti])} />
        <FactConfidence ti={fti} />
        <FactNll ti={fti} />
        <Legend items={[
          ...REGIMES.map((g) => ({ label: `${REGIME_LABEL[g]} (${FACTS[g].n} probes)`, color: REGIME_COLOR[g] })),
          { label: `all ${FACTS.all.n} probes`, color: 'var(--color-text-secondary)', dash: '5 3' },
        ]} />
        <Readouts>
          {REGIMES.map((g) => (
            <Readout key={g} id={`fact-${g}`} label={`${REGIME_LABEL[g]}: confidence / accuracy`}
              value={`${f3(FACTS[g].conf[fti])} / ${f3(FACTS[g].acc)}`} />
          ))}
          <Readout id="fact-nll-all" label="all probes: log loss at this τ" value={f3(FACTS.all.nll[fti])} />
          <Readout id="count-aware" label="log loss: τ = 1 / one τ* / τ* per regime"
            value={`${f3(ca.tau1)} / ${f3(ca.pooled)} / ${f3(ca.per_regime)}`} emphasis />
        </Readouts>
        <Note>
          Accuracy does not move with τ; confidence does. Facts seen twice or more are always right but underconfident, so
          their loss falls as τ falls; facts never seen are overconfident, so theirs falls as τ rises toward the uniform
          forecast. The pooled optimum ({f3(FACTS.all.tau_star)}) serves neither.
        </Note>
      </figure>
    </div>
  );
}
