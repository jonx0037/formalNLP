import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/contextual-embeddings-masked-lm/contextual_embeddings_masked_lm.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the views make. Change the .py, regenerate, never the reverse.
import payload from './data/masked-lm-lab.json';

/**
 * Masked-LM Laboratory — two figures for the contextual-embeddings-masked-lm topic.
 *   A. "Fill in the blank": an HMM sentence where any position can show [MASK], itself, or a substitute, with the
 *      exact Bayes posterior (Thm 2) beside the trained encoder's prediction; four filing windows with the encoder's
 *      guesses at a masked word and Thm 2's prediction at the visible word; and the four-cycle gap (Cor 2) for the
 *      HMM's own conditionals, the Bayes-optimal MLM's, and the encoder's.
 *   B. "One word, many vectors": the HMM polyseme's and kestrel's occurrences on their own principal axes, layer by
 *      layer, with Ethayarajh's measures (Def 3, Prop 2) and the probes; and the finance probes against their caps.
 */

type View = { exact: number[][]; model: number[][]; kl: number[] };
type Win = {
  kind: string; center: number; words: string[]; top: string[][]; top_p: number[][];
  masked_p: number[]; kept_p: number[]; formula_p: number[];
};
type Layer = { self_sim: number; mev: number; aniso: number; adj: number; mev_base: number; probe: number };

const CH = payload.channel as unknown as { rho: number; mask: number; random: number; kept: number; hmm_vocab: number; filing_vocab: number };
const HMM = payload.hmm as unknown as {
  tokens: number[]; names: string[]; substitute: number[]; views: Record<'mask' | 'kept' | 'random', View>;
  sense_a: (number | null)[]; fit: Record<string, number>;
};
const WINS = payload.filings as unknown as Win[];
const COMPAT = payload.compat as unknown as Record<'clean' | 'bayes' | 'model', number[]>;
const COMPAT_MED = payload.compat_median as unknown as Record<'clean' | 'bayes' | 'model', number>;
const CLOUDS = payload.clouds as unknown as {
  hmm: { xy: number[][]; sense_a: number[]; posterior: number[] };
  kestrel: { xy: number[][]; product: number[] };
};
const HMM_LAYERS = payload.hmm_layers as unknown as Layer[];
const HMM_PROBE = payload.hmm_probe as unknown as { static_cap: number; bayes: number; n: number };
const KES_LAYERS = payload.kestrel_layers as unknown as Layer[];
const ALIAS = payload.alias as unknown as Record<string, number>;
const NEWS = payload.news as unknown as {
  cap: number; control: number; n: number; layer0: number; layer1: number; layer2: number;
  words: Record<string, { n: number; good: number; cap: number; layer0: number; layer1: number; layer2: number }>;
};
const SW = payload.strong_weak as unknown as Record<string, number>;

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;

const N_H = HMM.tokens.length;
const V_H = CH.hmm_vocab;
const MODES = [
  { key: 'mask', label: 'shows [MASK]' },
  { key: 'kept', label: 'shows itself' },
  { key: 'random', label: 'shows a substitute' },
] as const;
const COMPAT_ROWS = [
  { key: 'clean', label: 'the HMM’s own conditionals', color: 'var(--color-badge-green-text)' },
  { key: 'bayes', label: 'the Bayes-optimal MLM', color: 'var(--color-badge-blue-text)' },
  { key: 'model', label: 'the trained encoder', color: 'var(--color-accent)' },
] as const;
const f3 = (v: number) => v.toFixed(3);

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

function Token({ active, onClick, children, data, mark }: {
  active: boolean; onClick: () => void; children: string; data: Record<string, string>; mark?: boolean;
}) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} {...data}
      style={{
        fontFamily: mono, fontSize: '0.75rem', padding: '0.1rem 0.3rem', borderRadius: '0.25rem', cursor: 'pointer',
        border: `1px solid ${mark ? 'var(--color-accent-secondary)' : 'var(--color-border)'}`,
        background: active ? 'var(--color-accent)' : mark ? 'var(--color-muted-bg)' : 'var(--color-surface)',
        color: active ? 'var(--color-bg)' : 'var(--color-text)', overflowWrap: 'anywhere', maxWidth: '100%',
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

function Legend({ items }: { items: { label: string; color: string; dash?: string; swatch?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="8" aria-hidden="true">
            {s.swatch
              ? <rect x="5" y="0" width="12" height="8" style={{ fill: s.color }} />
              : <line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />}
          </svg>
          {s.label}
        </span>
      ))}
    </div>
  );
}

function Slider({ id, label, value, min = 0, max, onChange, text }: {
  id: string; label: string; value: number; min?: number; max: number; onChange: (v: number) => void; text: string;
}) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', marginBottom: '0.45rem' }}>
      <label htmlFor={`mlm-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`mlm-lab-${id}`} type="range" min={min} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

/** A small table of rows of cells, as wrapping flex rows so it never overflows a phone screen. */
function Grid({ head, rows, id }: { head: string[]; rows: string[][]; id: string }) {
  const cell: CSSProperties = { flex: '1 1 0', minWidth: 0, padding: '0.2rem 0.3rem', overflowWrap: 'anywhere' };
  return (
    <div data-panel={id} style={{ fontFamily: sans, fontSize: '0.78rem', marginTop: '0.5rem' }}>
      <div style={{ display: 'flex', color: 'var(--color-text-secondary)', borderBottom: '1px solid var(--color-border)' }}>
        {head.map((h, i) => <div key={i} style={{ ...cell, flex: i === 0 ? '2 1 0' : '1 1 0' }}>{h}</div>)}
      </div>
      {rows.map((r, j) => (
        <div key={j} data-row={j} style={{ display: 'flex', borderBottom: '1px solid var(--color-border)' }}>
          {r.map((v, i) => <div key={i} style={{ ...cell, flex: i === 0 ? '2 1 0' : '1 1 0', fontFamily: i === 0 ? sans : mono }}>{v}</div>)}
        </div>
      ))}
    </div>
  );
}

/** Horizontal bars with optional reference lines, all in [0, 1]. */
function Bars({ id, rows, refs, title }: {
  id: string; title: string; rows: { label: string; value: number; color: string }[];
  refs: { label: string; value: number; dash?: string; band?: [number, number] }[];
}) {
  const W = 600, rowH = 26, M = { top: 22, right: 46, bottom: 28, left: 150 };
  const H = M.top + rows.length * rowH + M.bottom;
  const x = d3.scaleLinear().domain([0, 1]).range([0, W - M.left - M.right]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id}
      aria-label={`${title}: ${rows.map((r) => `${r.label} ${f3(r.value)}`).join(', ')}`}>
      <text x={8} y={14} style={titleText}>{title}</text>
      <g transform={`translate(${M.left},${M.top})`}>
        {refs.filter((r) => r.band).map((r) => (
          <rect key={`b${r.label}`} x={x(r.band![0])} width={x(r.band![1]) - x(r.band![0])} y={0} height={rows.length * rowH}
            style={{ fill: 'var(--color-badge-amber-bg)' }} />
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={x(v)} x2={x(v)} y1={0} y2={rows.length * rowH} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={x(v)} y={rows.length * rowH + 14} textAnchor="middle" style={axisText}>{v}</text>
          </g>
        ))}
        {rows.map((r, i) => (
          <g key={r.label}>
            <text x={-6} y={i * rowH + rowH / 2 + 3} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>{r.label}</text>
            <rect data-series={r.label} x={0} y={i * rowH + 4} width={x(r.value)} height={rowH - 8} style={{ fill: r.color }} />
            <text x={x(r.value) + 4} y={i * rowH + rowH / 2 + 3} style={{ ...axisText, fontFamily: mono }}>{f3(r.value)}</text>
          </g>
        ))}
        {refs.filter((r) => !r.band).map((r) => (
          <line key={r.label} x1={x(r.value)} x2={x(r.value)} y1={-4} y2={rows.length * rowH + 2}
            style={{ stroke: 'var(--color-text)', strokeWidth: 1.4, strokeDasharray: r.dash ?? '4 3' }} />
        ))}
      </g>
    </svg>
  );
}

const PW = 600, PH = 230, PM = { top: 22, right: 12, bottom: 34, left: 40 };
const IW = PW - PM.left - PM.right, IH = PH - PM.top - PM.bottom;

export default function MaskedLMLab() {
  const [viewA, setViewA] = useState<'hmm' | 'filings' | 'compat'>('hmm');
  const [pos, setPos] = useState(() => HMM.sense_a.findIndex((v) => v !== null));
  const [mode, setMode] = useState<'mask' | 'kept' | 'random'>('mask');
  const [win, setWin] = useState(0);
  const [wpos, setWpos] = useState(WINS[0].center);
  const [viewB, setViewB] = useState<'cloud' | 'probes'>('cloud');
  const [word, setWord] = useState<'hmm' | 'kestrel'>('hmm');
  const [layer, setLayer] = useState(1);
  const [probeSet, setProbeSet] = useState<'alias' | 'news' | 'polarity'>('alias');

  // ----- Figure A: the HMM sentence -----
  const view = HMM.views[mode];
  const exact = view.exact[pos], model = view.model[pos];
  const truth = HMM.tokens[pos];
  const shown = mode === 'mask' ? '[MASK]' : HMM.names[mode === 'kept' ? truth : HMM.substitute[pos]];
  const shownId = mode === 'kept' ? truth : mode === 'random' ? HMM.substitute[pos] : -1;
  const bx = d3.scaleBand<number>().domain(d3.range(V_H)).range([0, IW]).padding(0.18);
  const by = d3.scaleLinear().domain([0, Math.max(d3.max(exact)!, d3.max(model)!, 0.05) * 1.08]).range([IH, 0]).nice();

  // ----- Figure A: filings -----
  const W = WINS[win];

  // ----- Figure A: compatibility -----
  const cx = d3.scaleLinear().domain([-12, 1]).range([0, IW]);
  const jitter = (i: number) => ((i * 37) % 23) / 23 - 0.5;

  // ----- Figure B: cloud -----
  const cloud = word === 'hmm' ? CLOUDS.hmm : CLOUDS.kestrel;
  const flat = cloud.xy[layer];
  const pts = d3.range(flat.length / 2).map((i) => [flat[2 * i], flat[2 * i + 1]] as [number, number]);
  const groupA = word === 'hmm' ? CLOUDS.hmm.sense_a : CLOUDS.kestrel.product;
  const ext = Math.max(d3.max(pts, (p) => Math.max(Math.abs(p[0]), Math.abs(p[1])))!, 1e-6) * 1.12;
  const CS = 280;
  const sx = d3.scaleLinear().domain([-ext, ext]).range([0, CS]);
  const sy = d3.scaleLinear().domain([-ext, ext]).range([CS, 0]);
  const L = (word === 'hmm' ? HMM_LAYERS : KES_LAYERS)[layer];
  const colorA = word === 'hmm' ? 'var(--color-accent)' : 'var(--color-badge-blue-text)';
  const colorB = word === 'hmm' ? 'var(--color-badge-blue-text)' : 'var(--color-accent)';
  const labelA = word === 'hmm' ? 'sense A (regime a)' : 'ORG (Kestrel Turbine Services)';
  const labelB = word === 'hmm' ? 'sense B (regime b)' : 'PRODUCT (the Kestrel turbine)';
  const isA = (i: number) => (word === 'hmm' ? groupA[i] === 1 : groupA[i] === 0);

  return (
    <div data-lab="masked-lm" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Fill in the blank — the optimum of Theorem 2, the encoder, and the four-cycle test
        </figcaption>
        <Row>
          <Pill active={viewA === 'hmm'} onClick={() => setViewA('hmm')} data={{ 'data-view': 'hmm' }}>HMM sentence</Pill>
          <Pill active={viewA === 'filings'} onClick={() => setViewA('filings')} data={{ 'data-view': 'filings' }}>filing windows</Pill>
          <Pill active={viewA === 'compat'} onClick={() => setViewA('compat')} data={{ 'data-view': 'compat' }}>four-cycle gap</Pill>
        </Row>

        {viewA === 'hmm' && (
          <>
            <Row style={{ gap: '0.25rem' }}>
              {HMM.tokens.map((t, i) => (
                <Token key={i} active={i === pos} onClick={() => setPos(i)} data={{ 'data-pos': String(i) }} mark={HMM.sense_a[i] !== null}>
                  {HMM.names[t]}
                </Token>
              ))}
            </Row>
            <Row>
              {MODES.map((m) => (
                <Pill key={m.key} active={mode === m.key} onClick={() => setMode(m.key)} data={{ 'data-mode': m.key }}>{m.label}</Pill>
              ))}
            </Row>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="hmm-bars"
              aria-label={`At position ${pos + 1}, showing ${shown}: the exact Bayes posterior and the encoder's prediction over the 25 tokens`}>
              <g transform={`translate(${PM.left},${PM.top})`}>
                {by.ticks(4).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={IW} y1={by(v)} y2={by(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-5} y={by(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                {d3.range(V_H).map((k) => {
                  const w = bx.bandwidth() / 2;
                  return (
                    <g key={k}>
                      <rect data-series="exact" x={bx(k)!} width={w} y={by(exact[k])} height={IH - by(exact[k])} style={{ fill: 'var(--color-accent-secondary)' }} />
                      <rect data-series="model" x={bx(k)! + w} width={w} y={by(model[k])} height={IH - by(model[k])} style={{ fill: 'var(--color-accent)' }} />
                      <text x={bx(k)! + w} y={IH + 12} textAnchor="middle"
                        style={{ ...axisText, fontSize: '8.5px', fontWeight: k === truth || k === shownId ? 700 : 400, fill: k === truth ? 'var(--color-accent)' : 'var(--color-text-secondary)' }}>
                        {HMM.names[k]}
                      </text>
                    </g>
                  );
                })}
              </g>
              <text x={8} y={13} style={titleText}>{`position ${pos + 1} (true token ${HMM.names[truth]}) shows ${shown}`}</text>
              <text x={PW / 2} y={PH - 4} textAnchor="middle" style={axisText}>token (true token in bold color)</text>
            </svg>
            <Legend items={[
              { label: 'exact Bayes posterior (Theorem 2)', color: 'var(--color-accent-secondary)', swatch: true },
              { label: 'trained encoder', color: 'var(--color-accent)', swatch: true },
            ]} />
            <Readouts>
              <Readout id="kl" label="KL(exact ‖ encoder)" value={f3(view.kl[pos])} emphasis />
              <Readout id="exact-true" label="exact q(true token)" value={f3(exact[truth])} />
              <Readout id="model-true" label="encoder q(true token)" value={f3(model[truth])} />
              {HMM.sense_a[pos] !== null && <Readout id="sense" label="P(sense A | sentence)" value={f3(HMM.sense_a[pos]!)} />}
            </Readouts>
            <Note>
              At a [MASK] both bars are the context’s prediction. Shown its own token, the posterior shrinks toward it by
              Theorem 2 (ii), but keeps mass on the alternatives, because one selected position in ten shows a random token.
              Over the test set the encoder’s mean KL to the exact posterior is {f3(HMM.fit.kl)} nats, against an entropy of {f3(HMM.fit.entropy)}.
              The highlighted tokens are the polyseme w, once in each sense.
            </Note>
          </>
        )}

        {viewA === 'filings' && (
          <>
            <Row>
              {WINS.map((w, i) => (
                <Pill key={w.kind} active={i === win} onClick={() => { setWin(i); setWpos(w.center); }} data={{ 'data-window': String(i) }}>{w.kind}</Pill>
              ))}
            </Row>
            <Row style={{ gap: '0.25rem' }}>
              {W.words.map((t, i) => (
                <Token key={i} active={i === wpos} onClick={() => setWpos(i)} data={{ 'data-wpos': String(i) }} mark={i === W.center}>{t}</Token>
              ))}
            </Row>
            <Grid id="filing-top" head={[`masked “${W.words[wpos]}”: the encoder’s top ${W.top[wpos].length}`, 'probability']}
              rows={W.top[wpos].map((t, j) => [t === W.words[wpos] ? `${t} ✓` : t, f3(W.top_p[wpos][j])])} />
            <Readouts>
              <Readout id="masked-p" label="q(word | [MASK])" value={f3(W.masked_p[wpos])} />
              <Readout id="kept-p" label="q(word | word visible), encoder" value={f3(W.kept_p[wpos])} emphasis />
              <Readout id="formula-p" label="Theorem 2 from the [MASK] guess" value={f3(W.formula_p[wpos])} />
            </Readouts>
            <Note>
              With {CH.filing_vocab} tokens a random substitute almost never lands on a plausible word, so Theorem 2 says
              the optimum keeps nearly all its mass on a visible word. The encoder keeps less: visible positions carry
              little of its training signal. Its [MASK] guesses are where the context shows: in the “rose” windows, “rose”
              competes with “fell” and its synonyms, and the aspect before it is what a probe reads.
            </Note>
          </>
        )}

        {viewA === 'compat' && (
          <>
            <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="compat"
              aria-label={`log10 of the four-cycle gap over 200 pairs: medians ${COMPAT_ROWS.map((r) => `${r.label} ${COMPAT_MED[r.key]}`).join(', ')}`}>
              <g transform={`translate(${PM.left},${PM.top})`}>
                {[-12, -9, -6, -3, 0].map((v) => (
                  <g key={v}>
                    <line x1={cx(v)} x2={cx(v)} y1={0} y2={IH} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={cx(v)} y={IH + 13} textAnchor="middle" style={axisText}>{v === 0 ? '1' : `10^${v}`}</text>
                  </g>
                ))}
                {COMPAT_ROWS.map((r, j) => {
                  const yc = (j + 0.5) * (IH / 3);
                  return (
                    <g key={r.key} data-series={r.key}>
                      {COMPAT[r.key].map((v, i) => (
                        <circle key={i} cx={cx(v)} cy={yc + jitter(i) * (IH / 3) * 0.6} r={2.4} style={{ fill: r.color, fillOpacity: 0.55 }} />
                      ))}
                      <line x1={cx(Math.max(Math.log10(COMPAT_MED[r.key]), -12))} x2={cx(Math.max(Math.log10(COMPAT_MED[r.key]), -12))}
                        y1={yc - IH / 7} y2={yc + IH / 7} style={{ stroke: 'var(--color-text)', strokeWidth: 2 }} />
                    </g>
                  );
                })}
              </g>
              <text x={8} y={13} style={titleText}>|δ| over 200 random pairs (log scale; 10^−12 and below drawn at the left edge)</text>
            </svg>
            <Legend items={COMPAT_ROWS.map((r) => ({ label: r.label, color: r.color, swatch: true }))} />
            <Readouts>
              {COMPAT_ROWS.map((r) => <Readout key={r.key} id={`compat-${r.key}`} label={`median |δ|, ${r.label}`} value={String(COMPAT_MED[r.key])} />)}
            </Readouts>
            <Note>
              A joint’s conditionals give the same log-ratio along both paths around the square, so δ = 0 (Corollary 2): the
              HMM’s own conditionals sit at rounding. The trained encoder’s do not, so no distribution has them as its
              conditionals. Even the Bayes-optimal masked LM fails slightly, because at a [MASK] it reads a context that may
              itself be corrupted.
            </Note>
          </>
        )}
      </figure>

      <figure style={{ margin: '1.5rem 0 0' }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · One word, many vectors — occurrence clouds by layer, and what probes read from them
        </figcaption>
        <Row>
          <Pill active={viewB === 'cloud'} onClick={() => setViewB('cloud')} data={{ 'data-view': 'cloud' }}>the cloud</Pill>
          <Pill active={viewB === 'probes'} onClick={() => setViewB('probes')} data={{ 'data-view': 'probes' }}>finance probes</Pill>
        </Row>

        {viewB === 'cloud' && (
          <>
            <Row>
              <Pill active={word === 'hmm'} onClick={() => setWord('hmm')} data={{ 'data-word': 'hmm' }}>HMM polyseme w</Pill>
              <Pill active={word === 'kestrel'} onClick={() => setWord('kestrel')} data={{ 'data-word': 'kestrel' }}>kestrel (filings)</Pill>
            </Row>
            <Slider id="layer" label="layer" value={layer} max={2} onChange={setLayer} text={`layer ${layer}`} />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'flex-start' }}>
              <svg viewBox={`0 0 ${CS} ${CS}`} style={{ flex: '1 1 14rem', minWidth: 0, maxWidth: '22rem', height: 'auto' }} role="img" data-panel="cloud"
                aria-label={`${pts.length} occurrences of ${word === 'hmm' ? 'w' : 'kestrel'} at layer ${layer} on their top two principal axes, colored by sense`}>
                <rect x={0} y={0} width={CS} height={CS} style={{ fill: 'none', stroke: 'var(--color-border)' }} />
                {pts.map(([a, b], i) => (
                  <circle key={i} data-sense={isA(i) ? 'a' : 'b'} cx={sx(a)} cy={sy(b)} r={3.2}
                    style={{ fill: isA(i) ? colorA : colorB, fillOpacity: word === 'hmm' ? 0.2 + 0.7 * Math.abs(2 * CLOUDS.hmm.posterior[i] - 1) : 0.7, stroke: 'var(--color-bg)', strokeWidth: 0.6 }} />
                ))}
                <text x={6} y={14} style={titleText}>{L.mev === 1 && word === 'kestrel' ? 'one point: layer 0 is static' : 'top two principal axes'}</text>
              </svg>
              <div style={{ flex: '1 1 14rem', minWidth: 0 }}>
                <Legend items={[{ label: labelA, color: colorA, swatch: true }, { label: labelB, color: colorB, swatch: true }]} />
                <Readouts>
                  <Readout id="self-sim" label="self-similarity" value={f3(L.self_sim)} />
                  <Readout id="baseline" label="anisotropy baseline" value={f3(L.aniso)} />
                  <Readout id="adjusted" label="adjusted self-similarity" value={f3(L.adj)} emphasis />
                  <Readout id="mev" label="MEV (its baseline)" value={`${f3(L.mev)} (${f3(L.mev_base)})`} />
                  <Readout id="probe" label={word === 'hmm' ? 'sense probe' : 'ORG/PRODUCT probe, 5-fold'} value={f3(L.probe)} emphasis />
                  <Readout id="cap" label="static cap" value={f3(word === 'hmm' ? HMM_PROBE.static_cap : ALIAS.kestrel_majority)} />
                  {word === 'hmm' && <Readout id="bayes" label="exact Bayes accuracy" value={f3(HMM_PROBE.bayes)} />}
                </Readouts>
              </div>
            </div>
            <Note>
              {word === 'hmm'
                ? 'Shading is the exact sense posterior: faint points are occurrences whose context leaves the sense uncertain. Layer 0 is E[x] + P at the occurrence’s position, so it varies only with position and the probe sits at chance; one block of attention separates the senses almost to the Bayes accuracy.'
                : 'Every kestrel is read at the center of its own window, so layer 0 is one point for all 118 occurrences (MEV and self-similarity 1) and no probe can beat the majority. From layer 1 the company and the product are separate clouds.'}
            </Note>
          </>
        )}

        {viewB === 'probes' && (
          <>
            <Row>
              <Pill active={probeSet === 'alias'} onClick={() => setProbeSet('alias')} data={{ 'data-probes': 'alias' }}>company or product</Pill>
              <Pill active={probeSet === 'news'} onClick={() => setProbeSet('news')} data={{ 'data-probes': 'news' }}>the news a word reports</Pill>
              <Pill active={probeSet === 'polarity'} onClick={() => setProbeSet('polarity')} data={{ 'data-probes': 'polarity' }}>strong or weak</Pill>
            </Row>
            {probeSet === 'alias' && (
              <>
                <Bars id="bars-alias" title={`21 aliases, ${ALIAS.n} occurrences, leave one alias out (dashed: per-alias majority)`}
                  rows={[
                    { label: 'skip-gram (±2)', value: ALIAS.sgns, color: 'var(--color-text-secondary)' },
                    { label: 'layer 0 (static)', value: ALIAS.layer0, color: 'var(--color-text-secondary)' },
                    { label: 'layer 1', value: ALIAS.layer1, color: 'var(--color-accent)' },
                    { label: 'layer 2', value: ALIAS.layer2, color: 'var(--color-accent)' },
                  ]}
                  refs={[{ label: 'majority', value: ALIAS.majority }]} />
                <Note>The static representations sit at the majority; the contextual layers type the held-out alias from its neighbors.</Note>
              </>
            )}
            {probeSet === 'news' && (
              <>
                <Bars id="bars-news" title={`${NEWS.n} direction words and evaluative verbs, per-word 5-fold (dashed: static cap)`}
                  rows={[
                    { label: 'layer 0 (static)', value: NEWS.layer0, color: 'var(--color-text-secondary)' },
                    { label: 'layer 1', value: NEWS.layer1, color: 'var(--color-accent)' },
                    { label: 'layer 2', value: NEWS.layer2, color: 'var(--color-accent)' },
                    { label: 'shuffled control', value: NEWS.control, color: 'var(--color-badge-amber-text)' },
                  ]}
                  refs={[{ label: 'cap', value: NEWS.cap }]} />
                <Grid id="news-words" head={['word', 'n', 'good news', 'cap', 'layer 1']}
                  rows={Object.entries(NEWS.words).map(([w, v]) => [w, String(v.n), String(v.good), f3(v.cap), f3(v.layer1)])} />
                <Note>
                  “Costs rose” is bad news and “revenue rose” good. The direction words, with over a hundred occurrences each,
                  are read almost perfectly; the evaluative verbs, with 26 to 45 each, are not.
                </Note>
              </>
            )}
            {probeSet === 'polarity' && (
              <>
                <Bars id="bars-polarity" title="Topic 7’s polarity probe on 35 lexicon words (shaded: chance ± 2 SE)"
                  rows={[
                    { label: 'skip-gram (±2)', value: SW.sgns, color: 'var(--color-text-secondary)' },
                    { label: 'layer 0 (static)', value: SW.layer0, color: 'var(--color-text-secondary)' },
                    { label: 'layer 1', value: SW.layer1, color: 'var(--color-accent)' },
                    { label: 'layer 2', value: SW.layer2, color: 'var(--color-accent)' },
                  ]}
                  refs={[{ label: 'chance band', value: 0.5, band: [0.5 - 2 * SW.chance_se, 0.5 + 2 * SW.chance_se] }, { label: 'chance', value: 0.5 }]} />
                <Readouts>
                  <Readout id="vote8" label="neighbors’ vote, ±8 words" value={f3(SW.vote8)} />
                  <Readout id="vote32" label="neighbors’ vote, ±32 words" value={f3(SW.vote32)} />
                  <Readout id="vote-doc" label="whole document’s vote" value={f3(SW.vote_doc)} emphasis />
                </Readouts>
                <Note>
                  No representation leaves the chance band, because the text barely carries a word’s dictionary polarity: even a
                  whole document’s other lexicon words vote it right only {f3(SW.vote_doc)} of the time.
                </Note>
              </>
            )}
          </>
        )}
      </figure>
    </div>
  );
}
