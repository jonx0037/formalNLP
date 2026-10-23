import { useMemo, useState } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/neural-language-models/neural_language_models.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/neural-lm-lab.json';

/**
 * Neural Language Model Laboratory — two linked panels for the neural-language-models topic.
 *   A. "One sentence, four models": per-token surprisal under interpolated KN, the NPLM, the Elman RNN, and the
 *      RNN with dynamic evaluation. Token chips mark what training saw (trigram, bigram only, neither); an
 *      unknown word's spelling is the same hatched segment for every model (the spelling floor, Prop 6).
 *   B. "Gradients through time": for the token picked in A, how strongly its loss reaches back to every earlier
 *      state of the trained RNN, against Thm 5's bound; or, synthetically, ‖∂h_T/∂h_{T−m}‖ for W_h of spectral
 *      norm σ, tanh or linear, with the trained RNN's corpus curve; and the long-lag inset (Prop 4).
 */

type Model = 'kn' | 'nplm' | 'rnn' | 'rnn-dyn';
type Test = 'held' | 'edgar';
type Token = { w: string; oov: boolean; cls: 0 | 1 | 2; spell_bits: number | null; bits: Record<Model, number> };
type Sentence = {
  test: Test; index: number; words: string[]; chars: number; tokens: Token[];
  sensitivity: number[][]; sensitivity_bound: number[][]; step_norms: number[];
};
type Row = { nats: number; bpc: number; pp: number };
type Through = { sigmas: number[]; tanh: number[][]; linear: number[][]; rho: number[] };
type Decay = { lags: number[]; measured: number[]; bound: number[]; n: number[] };
type LagRow = { log10_reach: number; accuracy: number; test_loss: number; log10_cell_path: number | null };

const SENTENCES = payload.sentences as unknown as Sentence[];
const THROUGH = payload.through_time as unknown as Through;
const DECAY = payload.corpus_decay as unknown as Decay;
const LAG = payload.long_lag as unknown as { lag: number; models: Record<string, LagRow> };
const CORPUS = payload.corpus as unknown as Record<Test, Record<string, Row>>;
const SIGMA_MAX = payload.sigma_max as number;
const RHO = payload.rho as number;
const LZMA = payload.lzma_edgar_bpc as number;

const MODELS: { key: Model; label: string; color: string }[] = [
  { key: 'kn', label: 'Interpolated KN', color: 'var(--color-text-secondary)' },
  { key: 'nplm', label: 'NPLM', color: 'var(--color-accent-secondary)' },
  { key: 'rnn', label: 'Elman RNN', color: 'var(--color-accent)' },
  { key: 'rnn-dyn', label: 'RNN, dynamic evaluation', color: 'var(--color-badge-blue-text)' },
];
const CLASSES: { label: string; fill: string }[] = [
  { label: 'bigram never seen', fill: 'var(--color-badge-red-bg)' },
  { label: 'bigram seen, trigram not', fill: 'var(--color-badge-amber-bg)' },
  { label: 'trigram seen in training', fill: 'var(--color-badge-green-bg)' },
];
const LAG_LABEL: Record<string, string> = { elman: 'Elman RNN', 'lstm-1': 'LSTM, forget bias 1', 'lstm-3': 'LSTM, forget bias 3' };

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const fmt = d3.format(',');

function showWord(w: string): string {
  return w === '</s>' ? '⟨/s⟩' : w === '<s>' ? '⟨s⟩' : w;
}
/** Truncate a word for a narrow bar label. */
function short(w: string, n: number): string {
  return w.length > n ? `${w.slice(0, n - 1)}…` : w;
}
function sci(x: number): string {
  return x >= 0.01 && x < 1000 ? x.toPrecision(3) : x.toExponential(1);
}

function Pill({ active, onClick, children, data, color }: { active: boolean; onClick: () => void; children: string; data: Record<string, string>; color?: string }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} {...data}
      style={{
        fontFamily: sans, fontSize: '0.75rem', padding: '0.2rem 0.55rem', borderRadius: '999px', cursor: 'pointer',
        border: `1px solid ${color ?? 'var(--color-border)'}`,
        background: active ? (color ?? 'var(--color-accent)') : 'var(--color-surface)',
        color: active ? 'var(--color-bg)' : 'var(--color-text)',
      }}>
      {children}
    </button>
  );
}

function Readout({ label, value, id, emphasis }: { label: string; value: string; id: string; emphasis?: boolean }) {
  return (
    <div style={{ fontFamily: sans, fontSize: '0.8rem' }}>
      <div style={{ color: 'var(--color-text-secondary)' }}>{label}</div>
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)' }}>
        {value}
      </div>
    </div>
  );
}

// Panel A geometry: a word column, four thin bars per token, and a bits column.
const AW = 600, LABEL = 132, VALUE = 58, BAR = 5, GAP = 2, PAD = 5;
const X0 = LABEL, TRACK = AW - LABEL - VALUE;
// Panel B geometry.
const BW = 600, BH = 230, BM = { top: 18, right: 16, bottom: 52, left: 52 };
const SW = 600, SH = 240, SM = { top: 18, right: 16, bottom: 34, left: 52 };
const LW = 600, LROW = 26, LLABEL = 150, LVALUE = 150;

export default function NeuralLMLab() {
  const [sentIdx, setSentIdx] = useState(0);
  const [shown, setShown] = useState<Record<Model, boolean>>({ kn: true, nplm: true, rnn: true, 'rnn-dyn': true });
  const [tok, setTok] = useState(2);
  const [view, setView] = useState<'sentence' | 'synthetic'>('sentence');
  const [sigIdx, setSigIdx] = useState(THROUGH.sigmas.indexOf(1.5));
  const [mode, setMode] = useState<'tanh' | 'linear'>('tanh');

  const sent = SENTENCES[sentIdx];
  const t = Math.min(tok, sent.tokens.length - 1);
  const models = MODELS.filter((m) => shown[m.key]);

  const pickSentence = (i: number) => {
    setSentIdx(i);
    setTok(Math.min(tok, SENTENCES[i].tokens.length - 1));
    if (SENTENCES[i].test === 'edgar') setShown((s) => ({ ...s, 'rnn-dyn': true }));
  };
  const pickToken = (i: number) => {
    setTok(i);
    setView('sentence');
  };
  const toggleModel = (m: Model) => setShown((s) => {
    const next = { ...s, [m]: !s[m] };
    return Object.values(next).some(Boolean) ? next : s;
  });

  // Panel A — bars in bits on one linear scale per sentence (closed bits plus the shared spelling segment).
  const ax = useMemo(() => {
    const mx = d3.max(sent.tokens, (tk) => d3.max(MODELS, (m) => tk.bits[m.key])! + (tk.spell_bits ?? 0))!;
    return d3.scaleLinear().domain([0, mx * 1.02]).range([0, TRACK]);
  }, [sent]);
  const ROWH = models.length * (BAR + GAP) - GAP + 2 * PAD;
  const AH = sent.tokens.length * ROWH + 4;
  const spell = d3.sum(sent.tokens, (tk) => tk.spell_bits ?? 0);
  const totals = MODELS.map((m) => ({ ...m, closed: d3.sum(sent.tokens, (tk) => tk.bits[m.key]) }));
  const whole = CORPUS[sent.test];

  // Panel B, sentence view — ‖∂ℓ_t/∂h_k‖ for k = 0 … t (h_k has just read word k; h_0 has read ⟨s⟩).
  const sens = sent.sensitivity[t];
  const bnd = sent.sensitivity_bound[t];
  const readWords = ['<s>', ...sent.words, '</s>'].slice(0, t + 1);
  const bx = d3.scaleBand<number>().domain(d3.range(t + 1)).range([0, BW - BM.left - BM.right]).padding(0.25);
  const blo = Math.min(d3.min(sens)!, d3.min(bnd)!), bhi = Math.max(d3.max(sens)!, d3.max(bnd)!);
  const by = d3.scaleLog().domain([blo / 1.6, bhi * 1.6]).range([BH - BM.top - BM.bottom, 0]);
  const byTicks = by.ticks().filter((v) => Number.isInteger(Math.log10(v)));

  // Panel B, synthetic view — median log₁₀ ‖∂h_T/∂h_{T−m}‖ against lag m, for each σ.
  const curves = mode === 'tanh' ? THROUGH.tanh : THROUGH.linear;
  const sigma = THROUGH.sigmas[sigIdx];
  const lags = d3.range(1, curves[0].length + 1);
  const sx = d3.scaleLinear().domain([1, lags.length]).range([0, SW - SM.left - SM.right]);
  const flat = curves.flat();
  const sy = d3.scaleLinear().domain([Math.min(-3, d3.min(flat)!), Math.max(1, d3.max(flat)!) + 0.5]).range([SH - SM.top - SM.bottom, 0]);
  const line = d3.line<number>().x((_, i) => sx(i + 1)).y((v) => sy(v));
  const boundY = (m: number) => m * Math.log10(sigma);
  const [ylo, yhi] = sy.domain();
  const boundPts = lags.filter((m) => boundY(m) >= ylo && boundY(m) <= yhi);
  const decayLine = d3.line<number>().x((_, i) => sx(DECAY.lags[i])).y((v) => sy(Math.log10(v)));

  // The long-lag inset — log₁₀ of the gradient reaching the remembered bit at initialization.
  const lagKeys = Object.keys(LAG.models);
  const lx = d3.scaleLinear().domain([-11, 0]).range([0, LW - LLABEL - LVALUE]);

  return (
    <div data-lab="neural-lm" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · One sentence, four models — bits per token, and what training saw
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.4rem' }}>
          {SENTENCES.map((s, i) => (
            <Pill key={i} active={i === sentIdx} onClick={() => pickSentence(i)} data={{ 'data-sentence': String(i) }}>
              {`${s.test === 'edgar' ? 'EDGAR: ' : ''}${s.words.slice(0, 3).join(' ')} …`}
            </Pill>
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          {MODELS.map((m) => (
            <Pill key={m.key} active={shown[m.key]} onClick={() => toggleModel(m.key)} color={m.color} data={{ 'data-model': m.key }}>
              {m.label}
            </Pill>
          ))}
        </div>
        <svg viewBox={`0 0 ${AW} ${AH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="tokens"
          aria-label="Bits each model spends on each token of the sentence; hatched segments are the spelling of unknown words, the same for every model">
          <defs>
            <pattern id="nlm-hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="5" style={{ stroke: 'var(--color-accent-secondary)', strokeWidth: 2 }} />
            </pattern>
          </defs>
          {sent.tokens.map((tk, i) => {
            const y = 2 + i * ROWH;
            const picked = i === t;
            return (
              <g key={i} data-token-row={i} data-cls={tk.cls} onClick={() => pickToken(i)} style={{ cursor: 'pointer' }}>
                <rect x={0} y={y} width={AW} height={ROWH - 1} rx={3}
                  style={{ fill: picked ? 'var(--color-muted-bg)' : 'transparent', stroke: picked ? 'var(--color-accent)' : 'none' }} />
                <rect x={4} y={y + ROWH / 2 - 5} width={10} height={10} rx={2}
                  style={{ fill: CLASSES[tk.cls].fill, stroke: 'var(--color-border)' }} />
                <text x={LABEL - 8} y={y + ROWH / 2 + 4} textAnchor="end"
                  style={{ fontFamily: mono, fontSize: '11.5px', fill: tk.oov ? 'var(--color-accent-secondary)' : 'var(--color-text)', fontWeight: picked ? 700 : 400 }}>
                  {showWord(tk.w)}
                </text>
                {models.map((m, j) => {
                  const by0 = y + PAD + j * (BAR + GAP);
                  const w = Math.max(1, ax(tk.bits[m.key]));
                  return (
                    <g key={m.key} data-bar={m.key}>
                      <rect x={X0} y={by0} width={w} height={BAR} style={{ fill: m.color }} />
                      {tk.spell_bits !== null && (
                        <rect x={X0 + w} y={by0} width={ax(tk.spell_bits)} height={BAR} style={{ fill: 'url(#nlm-hatch)' }} />
                      )}
                    </g>
                  );
                })}
                <text x={AW - 4} y={y + ROWH / 2 + 4} textAnchor="end"
                  style={{ fontFamily: mono, fontSize: '10.5px', fill: 'var(--color-text-secondary)' }}>
                  {tk.oov ? 'spelled' : ''}
                </text>
              </g>
            );
          })}
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem 1rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', margin: '0.3rem 0 0.5rem' }}>
          {CLASSES.map((c, i) => (
            <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: c.fill, border: '1px solid var(--color-border)', display: 'inline-block' }} />
              {c.label}
            </span>
          ))}
          <span>gold words are unknown and spelled letter by letter (hatched)</span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem' }}>
          {totals.map((m) => (
            <Readout key={m.key} id={`total-${m.key}`} label={`${m.label}, whole sentence`} emphasis={m.key === 'rnn'}
              value={`${m.closed.toFixed(1)} bits${spell ? ` + ${spell.toFixed(1)}` : ''}`} />
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>
          <Readout id="token" label="picked token" value={showWord(sent.tokens[t].w)} />
          {MODELS.map((m) => (
            <Readout key={m.key} id={`token-${m.key}`} label={m.label} value={`${sent.tokens[t].bits[m.key].toFixed(2)} bits`} />
          ))}
        </div>
        <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
          Click a token to see, in panel B, how far back its loss reaches. Totals are closed-vocabulary bits; the second
          number is the spelling, which every model pays. On the whole {sent.test === 'held' ? 'held-out' : 'EDGAR'} text, in
          bits per character: {MODELS.map((m) => `${m.label} ${whole[m.key].bpc.toFixed(4)}`).join(', ')}
          {sent.test === 'edgar' ? `; lzma, given the training text, ${LZMA.toFixed(3)}` : ''}.
        </p>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Gradients through time — products of Jacobians, against their bound
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          <Pill active={view === 'sentence'} onClick={() => setView('sentence')} data={{ 'data-view': 'sentence' }}>this sentence, trained RNN</Pill>
          <Pill active={view === 'synthetic'} onClick={() => setView('synthetic')} data={{ 'data-view': 'synthetic' }}>synthetic W_h of spectral norm σ</Pill>
        </div>

        {view === 'sentence' ? (
          <>
            <svg viewBox={`0 0 ${BW} ${BH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="sensitivity"
              aria-label="How strongly the picked token's loss depends on each earlier hidden state, on a log scale, with the bound from per-step norms">
              <g transform={`translate(${BM.left},${BM.top})`}>
                {byTicks.map((v) => (
                  <g key={v}>
                    <line x1={0} x2={BW - BM.left - BM.right} y1={by(v)} y2={by(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={by(v) + 3} textAnchor="end" style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' }}>
                      {d3.format('~g')(v)}
                    </text>
                  </g>
                ))}
                {sens.map((v, k) => (
                  <g key={k} data-state={k}>
                    <rect x={bx(k)} y={by(v)} width={bx.bandwidth()} height={BH - BM.top - BM.bottom - by(v)}
                      style={{ fill: 'var(--color-accent)', fillOpacity: 0.85 }} />
                    <circle cx={bx(k)! + bx.bandwidth() / 2} cy={by(bnd[k])} r={3}
                      style={{ fill: 'var(--color-surface)', stroke: 'var(--color-accent-secondary)', strokeWidth: 1.5 }} />
                    <text x={bx(k)! + bx.bandwidth() / 2} y={BH - BM.top - BM.bottom + 13} textAnchor="middle"
                      style={{ fontFamily: mono, fontSize: '9.5px', fill: 'var(--color-text)' }}>
                      {short(showWord(readWords[k]), t > 7 ? 7 : 9)}
                    </text>
                    {k > 0 && (
                      <text x={bx(k)! + bx.bandwidth() / 2} y={BH - BM.top - BM.bottom + 27} textAnchor="middle"
                        style={{ fontFamily: mono, fontSize: '9px', fill: 'var(--color-text-secondary)' }}>
                        {`×${sent.step_norms[k].toFixed(2)}`}
                      </text>
                    )}
                  </g>
                ))}
                <path d={d3.line<number>().x((_, k) => bx(k)! + bx.bandwidth() / 2).y((v) => by(v))(bnd)!}
                  style={{ fill: 'none', stroke: 'var(--color-accent-secondary)', strokeWidth: 1.5, strokeDasharray: '5 3' }} data-series="bound" />
              </g>
              <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
                {`‖∂ℓ/∂h_k‖ for the loss at “${showWord(sent.tokens[t].w)}” (log scale)`}
              </text>
              <text x={BW - 8} y={12} textAnchor="end" style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent-secondary)' }}>
                ○ bound: product of per-step norms
              </text>
              <text x={BM.left} y={BH - 4} style={{ fontFamily: sans, fontSize: '9.5px', fill: 'var(--color-text-secondary)' }}>
                state h_k has just read the word under it; ×n is the norm of the step into it
              </text>
            </svg>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.25rem' }}>
              <Readout id="reach-now" label="at the token's own state" value={sci(sens[t])} />
              <Readout id="reach-start" label="at ⟨s⟩, the first state" value={sci(sens[0])} emphasis />
              <Readout id="bound-start" label="Theorem 5's bound there" value={sci(bnd[0])} />
              <Readout id="sigma" label="σ_max(W_h), ρ(W_h)" value={`${SIGMA_MAX.toFixed(2)}, ${RHO.toFixed(2)}`} />
            </div>
            <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
              Every per-step factor exceeds 1, so the bound only grows going back. The measured gradient does not follow it:
              the factors' expanding directions do not line up, and the product shrinks or holds steady while the bound explodes.
            </p>
          </>
        ) : (
          <>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.4rem' }}>
              <Pill active={mode === 'tanh'} onClick={() => setMode('tanh')} data={{ 'data-mode': 'tanh' }}>tanh</Pill>
              <Pill active={mode === 'linear'} onClick={() => setMode('linear')} data={{ 'data-mode': 'linear' }}>linear (no squashing)</Pill>
            </div>
            <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem', fontFamily: sans, fontSize: '0.85rem', marginBottom: '0.25rem' }}>
              <span>spectral norm σ = <strong data-readout="sigma-synth">{sigma}</strong></span>
              <input type="range" min={0} max={THROUGH.sigmas.length - 1} step={1} value={sigIdx} aria-label="Spectral norm of the recurrent matrix"
                onChange={(e) => setSigIdx(parseInt(e.target.value, 10))} style={{ flex: '1 1 8rem', minWidth: 0, accentColor: 'var(--color-accent)' }} />
            </label>
            <svg viewBox={`0 0 ${SW} ${SH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="through-time"
              aria-label="Median log norm of the Jacobian product against the lag, for recurrent matrices of each spectral norm; the selected one is bold, with its bound dashed">
              <g transform={`translate(${SM.left},${SM.top})`}>
                {sy.ticks(6).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={SW - SM.left - SM.right} y1={sy(v)} y2={sy(v)}
                      style={{ stroke: v === 0 ? 'var(--color-text-secondary)' : 'var(--color-border)', strokeWidth: v === 0 ? 1 : 0.5 }} />
                    <text x={-6} y={sy(v) + 3} textAnchor="end" style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' }}>
                      {`10^${v}`}
                    </text>
                  </g>
                ))}
                {curves.map((c, i) => (
                  <path key={i} d={line(c)!} data-sigma={THROUGH.sigmas[i]}
                    style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: i === sigIdx ? 2.5 : 1, strokeOpacity: i === sigIdx ? 1 : 0.25 }} />
                ))}
                {boundPts.length > 1 && (
                  <path d={d3.line<number>().x((m) => sx(m)).y((m) => sy(boundY(m)))(boundPts)!} data-series="sigma-bound"
                    style={{ fill: 'none', stroke: 'var(--color-accent-secondary)', strokeWidth: 1.5, strokeDasharray: '5 3' }} />
                )}
                {mode === 'tanh' && (
                  <path d={decayLine(DECAY.measured)!} data-series="trained"
                    style={{ fill: 'none', stroke: 'var(--color-badge-blue-text)', strokeWidth: 2 }} />
                )}
                {[1, 10, 20, 30, 40, 50].map((m) => (
                  <text key={m} x={sx(m)} y={SH - SM.top - SM.bottom + 14} textAnchor="middle"
                    style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' }}>{m}</text>
                ))}
              </g>
              <text x={SW / 2} y={SH - 4} textAnchor="middle" style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' }}>lag m (steps back)</text>
              <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>median ‖∂h_T/∂h_(T−m)‖</text>
              <text x={SW - 8} y={12} textAnchor="end" style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent-secondary)' }}>
                {mode === 'tanh' ? 'dashed: σ^m · blue: trained RNN, held-out median' : 'dashed: σ^m'}
              </text>
            </svg>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.25rem' }}>
              <Readout id="rho-synth" label="median spectral radius ρ" value={THROUGH.rho[sigIdx].toFixed(3)} />
              <Readout id="lag50" label="median norm at lag 50" value={`10^${curves[sigIdx][lags.length - 1].toFixed(1)}`} emphasis />
              <Readout id="bound50" label="bound σ^50" value={`10^${boundY(50).toFixed(1)}`} />
            </div>
            <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
              Below σ = 1 the bound forces vanishing. Above it, growth is allowed but tanh rarely delivers it. Without squashing,
              the spectral radius decides, and for these random matrices ρ is about σ/2, so linear products explode only past σ ≈ 2.
              The trained RNN has σ_max = {SIGMA_MAX.toFixed(2)}, yet its gradients fade within about a dozen words.
            </p>
          </>
        )}

        <svg viewBox={`0 0 ${LW} ${lagKeys.length * LROW + 26}`} style={{ width: '100%', height: 'auto', marginTop: '0.75rem' }} role="img" data-panel="long-lag"
          aria-label={`Gradient reaching a bit remembered across ${LAG.lag} steps, at initialization, and test accuracy after training`}>
          <text x={0} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fontWeight: 600, fill: 'var(--color-text)' }}>
            {`Remember one bit across ${LAG.lag} noise symbols — gradient reaching it at initialization (log₁₀), accuracy after training`}
          </text>
          {lagKeys.map((k, i) => {
            const r = LAG.models[k];
            const y = 22 + i * LROW;
            return (
              <g key={k} data-lag-row={k}>
                <text x={LLABEL - 8} y={y + LROW / 2 + 1} textAnchor="end" style={{ fontFamily: sans, fontSize: '11px', fill: 'var(--color-text)' }}>{LAG_LABEL[k] ?? k}</text>
                <rect x={LLABEL} y={y + 4} width={Math.max(1, lx(r.log10_reach))} height={LROW - 12} rx={3}
                  style={{ fill: r.accuracy > 0.95 ? 'var(--color-accent)' : 'var(--color-muted-border)' }} />
                {r.log10_cell_path !== null && r.log10_cell_path > -11 && (
                  <line x1={LLABEL + lx(r.log10_cell_path)} x2={LLABEL + lx(r.log10_cell_path)} y1={y + 1} y2={y + LROW - 5}
                    style={{ stroke: 'var(--color-accent-secondary)', strokeWidth: 2 }} />
                )}
                <text x={LW - 4} y={y + LROW / 2 + 1} textAnchor="end" style={{ fontFamily: mono, fontSize: '10.5px', fill: 'var(--color-text-secondary)' }}>
                  {`10^${r.log10_reach.toFixed(1)} · acc ${r.accuracy.toFixed(3)}`}
                </text>
              </g>
            );
          })}
        </svg>
        <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.25rem 0 0' }}>
          Medians over 12 initializations; gold ticks mark the LSTM's cell path σ(b_f)^{LAG.lag} (Proposition 4). Only the network
          whose gradient reaches the bit learns to report it.
        </p>
      </figure>
    </div>
  );
}
