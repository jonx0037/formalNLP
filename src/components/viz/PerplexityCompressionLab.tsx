import { useMemo, useState } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/perplexity-cross-entropy/perplexity_cross_entropy.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/perplexity-compression-lab.json';

/**
 * Perplexity & Compression Laboratory — two linked panels for the perplexity topic.
 *   A. "Arithmetic-code a sentence": a cascade of nested intervals. Each row is the current interval,
 *      stretched to full width and cut into the slices of the likeliest next symbols; the chosen slice is
 *      zoomed into the next row. Unknown words switch the alphabet to letters (the speller). The ticker
 *      shows the codeword bits already settled; the final codeword is within 2 bits of −log₂ Q (Thm 4).
 *   B. "Pick a unit": eight models on the held-out (or EDGAR) text, ranked per token, per character, or
 *      per word token (Thm 5), and the unknown-word cutoff: closed perplexity falls as the honest code
 *      grows (Prop 5). The test-set toggle and the sentence picker are linked.
 */

type Smoother = 'laplace' | 'ikn' | 'mkn';
type Step = { s: [string, 'w' | 'c', number, number, number, number]; top: [string, number, number][] };
type Trace = { steps: Step[]; length: number; model_bits: number; codeword: string; shannon_bits: number };
type Sentence = { test: Test; words: string[]; chars: number; oov: string[]; by_smoother: Record<Smoother, Trace> };
type Test = 'held' | 'edgar';
type UnitRow = {
  model: string; is_code: boolean; tokens: number; tokens_per_char: number; nats_per_token: number;
  pp_token: number; bpc: number; pp_word: number;
};
type CutoffRow = {
  k: number | null; V: number; oov_share: number; closed_nats_per_token: number; closed_pp: number;
  closed_bpc: number; open_bpc: number;
};
type Metric = 'token' | 'char' | 'word';

const SENTENCES = payload.sentences as unknown as Sentence[];
const UNITS = payload.units as unknown as Record<Test, UnitRow[]>;
const CUTOFFS = payload.cutoffs as unknown as Record<Test, CutoffRow[]>;
const CODES = payload.codes as unknown as Record<Test, { length: number; model_bits: number; chars: number; shannon_bits: number }>;
const V_SIZE = payload.V as number;

const SMOOTHERS: { key: Smoother; label: string }[] = [
  { key: 'laplace', label: 'Laplace' },
  { key: 'ikn', label: 'Interpolated KN' },
  { key: 'mkn', label: 'Modified KN' },
];
const TESTS: { key: Test; label: string }[] = [
  { key: 'held', label: 'in domain (held out)' },
  { key: 'edgar', label: 'EDGAR MD&A' },
];
const METRICS: { key: Metric; label: string }[] = [
  { key: 'token', label: 'perplexity per token' },
  { key: 'char', label: 'bits per character' },
  { key: 'word', label: 'perplexity per word token' },
];
const MODEL_LABEL: Record<string, string> = {
  'char-3': 'character 3-gram', 'char-5': 'character 5-gram', 'char-7': 'character 7-gram',
  'bpe-100': 'BPE |V| = 100, 5-gram', 'bpe-400': 'BPE |V| = 400, 5-gram', 'bpe-1870': 'BPE |V| = 1,870, 5-gram',
  word: 'word trigram, open', 'word-closed': 'word trigram, ⟨unk⟩ closed',
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const fmt = d3.format(',');

/** How a symbol reads in a row: words as words, `</s>` as the sentence end (word mode) or end of spelling. */
function showSymbol(sym: string, mode: 'w' | 'c'): string {
  if (sym === '<unk>') return '⟨unk⟩';
  if (sym === '</s>') return mode === 'w' ? '⟨/s⟩' : '⟨end⟩';
  return mode === 'c' ? `‘${sym}’` : sym;
}

function metricOf(r: UnitRow, m: Metric): number {
  return m === 'token' ? r.pp_token : m === 'char' ? r.bpc : r.pp_word;
}
function metricText(v: number, m: Metric): string {
  if (m === 'char') return `${v.toFixed(4)} bpc`;
  return v >= 1e6 ? v.toExponential(2) : v >= 100 ? fmt(Math.round(v)) : v.toFixed(3);
}

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
    <div style={{ fontFamily: sans, fontSize: '0.8rem' }}>
      <div style={{ color: 'var(--color-text-secondary)' }}>{label}</div>
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)' }}>
        {value}
      </div>
    </div>
  );
}

// Panel A geometry: a symbol column, the interval track, and a bits column; rows joined by zoom wedges.
const AW = 600, LABEL = 118, VALUE = 84, ROW = 30, WEDGE = 20, WINDOW = 6;
const X0 = LABEL, X1 = AW - VALUE, TRACK = X1 - X0;

// Panel B geometry.
const BW = 600, BLABEL = 196, BVALUE = 96, BROW = 24;
const CW = 600, CH = 210, CM = { top: 16, right: 52, bottom: 34, left: 52 };

export default function PerplexityCompressionLab() {
  const [sentIdx, setSentIdx] = useState(0);
  const [smoother, setSmoother] = useState<Smoother>('ikn');
  const [step, setStep] = useState(1);
  const [test, setTest] = useState<Test>('held');
  const [metric, setMetric] = useState<Metric>('token');
  const [cutIdx, setCutIdx] = useState(0);

  const sent = SENTENCES[sentIdx];
  const trace = sent.by_smoother[smoother];
  const n = trace.steps.length;
  const k = Math.min(step, n);
  const cur = trace.steps[k - 1];

  const pickSentence = (i: number) => {
    setSentIdx(i);
    setStep(1);
    setTest(SENTENCES[i].test);
  };
  const pickTest = (t: Test) => {
    setTest(t);
    if (SENTENCES[sentIdx].test !== t) pickSentence(SENTENCES.findIndex((s) => s.test === t));
  };

  // Panel A — the rows on screen: the last WINDOW coded symbols.
  const first = Math.max(0, k - WINDOW);
  const rows = trace.steps.slice(first, k);
  const AH = rows.length * (ROW + WEDGE) + 4;

  // Panel B — models sorted by the chosen unit (lower is better).
  const unitRows = useMemo(
    () => [...UNITS[test]].sort((a, b) => metricOf(a, metric) - metricOf(b, metric)),
    [test, metric],
  );
  const BH = unitRows.length * BROW + 8;
  const bx = useMemo(() => {
    const vals = UNITS[test].map((r) => metricOf(r, metric));
    return metric === 'char'
      ? d3.scaleLinear().domain([0, d3.max(vals)! * 1.05]).range([0, BW - BLABEL - BVALUE])
      : d3.scaleLog().domain([1, d3.max(vals)! * 1.5]).range([0, BW - BLABEL - BVALUE]);
  }, [test, metric]);

  const cut = CUTOFFS[test];
  const cx = d3.scalePoint<number>().domain(cut.map((_, i) => i)).range([0, CW - CM.left - CM.right]).padding(0.3);
  const cyL = d3.scaleLog().domain([1, d3.max(cut, (r) => r.closed_pp)! * 1.3]).range([CH - CM.top - CM.bottom, 0]);
  const cyR = d3.scaleLinear().domain([0, d3.max(cut, (r) => r.open_bpc)! * 1.15]).range([CH - CM.top - CM.bottom, 0]);
  // Log-axis labels: 1–2–5 per decade over a narrow range, powers of ten over a wide one (EDGAR spans four decades).
  const wide = Math.log10(cyL.domain()[1]) > 2;
  const ticksL = cyL.ticks().filter((t) => (wide ? Number.isInteger(Math.log10(t)) : /^[125]0*$/.test(String(t))));
  const lineL = d3.line<CutoffRow>().x((_, i) => cx(i)!).y((r) => cyL(r.closed_pp));
  const lineR = d3.line<CutoffRow>().x((_, i) => cx(i)!).y((r) => cyR(r.open_bpc));
  const cr = cut[cutIdx];
  const kLabel = (r: CutoffRow) => (r.k === null ? 'all' : String(r.k));
  const code = CODES[test];

  return (
    <div data-lab="perplexity-compression" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Arithmetic-code a sentence — each symbol keeps its slice of the interval
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.4rem' }}>
          {SENTENCES.map((s, i) => (
            <Pill key={i} active={i === sentIdx} onClick={() => pickSentence(i)} data={{ 'data-sentence': String(i) }}>
              {`${s.test === 'edgar' ? 'EDGAR: ' : ''}${s.words.slice(0, 3).join(' ')} …`}
            </Pill>
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          {SMOOTHERS.map((s) => (
            <Pill key={s.key} active={s.key === smoother} onClick={() => setSmoother(s.key)} data={{ 'data-smoother': s.key }}>{s.label}</Pill>
          ))}
        </div>
        <p style={{ fontFamily: mono, fontSize: '0.8rem', margin: '0 0 0.5rem', lineHeight: 1.5 }} data-readout="sentence">
          {sent.words.map((w, i) => (
            <span key={i} style={{ color: sent.oov.includes(w) ? 'var(--color-accent-secondary)' : 'var(--color-text)' }}>
              {w}{i < sent.words.length - 1 ? ' ' : ''}
            </span>
          ))}
          <span style={{ fontFamily: sans, color: 'var(--color-text-secondary)' }}>{`  (${sent.chars} characters${sent.oov.length ? `; unknown words in gold` : ''})`}</span>
        </p>
        <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.6rem', fontFamily: sans, fontSize: '0.85rem', marginBottom: '0.5rem' }}>
          <button type="button" onClick={() => setStep(Math.max(1, k - 1))} aria-label="Previous symbol" data-control="prev"
            style={{ fontFamily: sans, border: '1px solid var(--color-border)', background: 'var(--color-surface)', color: 'var(--color-text)', borderRadius: '0.4rem', cursor: 'pointer', padding: '0 0.5rem' }}>◀</button>
          <input type="range" min={1} max={n} step={1} value={k} aria-label="Symbols coded so far"
            onChange={(e) => setStep(parseInt(e.target.value, 10))} style={{ flex: '1 1 6rem', minWidth: 0, accentColor: 'var(--color-accent)' }} />
          <button type="button" onClick={() => setStep(Math.min(n, k + 1))} aria-label="Next symbol" data-control="next"
            style={{ fontFamily: sans, border: '1px solid var(--color-border)', background: 'var(--color-surface)', color: 'var(--color-text)', borderRadius: '0.4rem', cursor: 'pointer', padding: '0 0.5rem' }}>▶</button>
          <span style={{ minWidth: '6.5rem' }}>symbol <strong data-readout="step">{k}</strong> of {n}</span>
        </label>
        <svg viewBox={`0 0 ${AW} ${AH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="intervals"
          aria-label="Nested intervals of the arithmetic code: each row is the current interval cut into slices for the likeliest next symbols, with the chosen slice zoomed into the next row">
          {rows.map((st, j) => {
            const [sym, mode, q, lo] = st.s;
            const prevBits = first + j === 0 ? 0 : trace.steps[first + j - 1].s[4];
            const y = 2 + j * (ROW + WEDGE);
            const cx0 = X0 + lo * TRACK, cw = Math.max(1.5, q * TRACK);
            const isLast = j === rows.length - 1;
            return (
              <g key={first + j} data-step-row={first + j} data-mode={mode}>
                <rect x={X0} y={y} width={TRACK} height={ROW} rx={3}
                  style={{ fill: mode === 'c' ? 'var(--color-definition-bg)' : 'var(--color-muted-bg)', stroke: 'var(--color-border)' }} />
                {st.top.map(([alt, aq, alo]) => {
                  const ax = X0 + alo * TRACK, aw = Math.max(1, aq * TRACK);
                  return (
                    <g key={alt}>
                      <rect x={ax} y={y + 3} width={aw} height={ROW - 6}
                        style={{ fill: 'var(--color-surface)', stroke: 'var(--color-muted-border)', strokeWidth: 1 }} />
                      {aw >= 38 && alt !== sym && (
                        <text x={ax + aw / 2} y={y + ROW / 2 + 4} textAnchor="middle"
                          style={{ fontFamily: mono, fontSize: '10px', fill: 'var(--color-text-secondary)' }}>
                          {showSymbol(alt, mode)}
                        </text>
                      )}
                    </g>
                  );
                })}
                <rect x={cx0} y={y + 1} width={cw} height={ROW - 2} data-chosen={sym}
                  style={{ fill: 'var(--color-accent)', fillOpacity: 0.85, stroke: 'var(--color-accent)' }} />
                {cw >= 44 && (
                  <text x={cx0 + cw / 2} y={y + ROW / 2 + 4} textAnchor="middle"
                    style={{ fontFamily: mono, fontSize: '10.5px', fontWeight: 600, fill: 'var(--color-bg)' }}>
                    {showSymbol(sym, mode)}
                  </text>
                )}
                <text x={LABEL - 8} y={y + ROW / 2 + 4} textAnchor="end"
                  style={{ fontFamily: mono, fontSize: '11.5px', fill: mode === 'c' ? 'var(--color-accent-secondary)' : 'var(--color-text)' }}>
                  {showSymbol(sym, mode)}
                </text>
                <text x={X1 + 8} y={y + ROW / 2 + 4}
                  style={{ fontFamily: mono, fontSize: '11px', fill: 'var(--color-text-secondary)' }}>
                  {`${(st.s[4] - prevBits).toFixed(2)} bits`}
                </text>
                {!isLast && (
                  <polygon points={`${cx0},${y + ROW} ${cx0 + cw},${y + ROW} ${X1},${y + ROW + WEDGE} ${X0},${y + ROW + WEDGE}`}
                    style={{ fill: 'var(--color-accent)', fillOpacity: 0.12 }} />
                )}
              </g>
            );
          })}
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.4rem' }}>
          <Readout id="bits" label="−log₂ Q so far" value={`${cur.s[4].toFixed(3)} bits`} emphasis />
          <Readout id="settled" label="codeword bits already settled" value={String(cur.s[5])} />
          <Readout id="codeword-length" label="final codeword (whole sentence)"
            value={`${trace.length} bits vs −log₂ Q = ${trace.model_bits.toFixed(2)}`} />
          <Readout id="shannon" label="symbol by symbol (Shannon codes)" value={`${trace.shannon_bits} bits`} />
        </div>
        <div data-readout="codeword" aria-label="Codeword bits; the settled prefix is highlighted"
          style={{ fontFamily: mono, fontSize: '0.8rem', wordBreak: 'break-all', marginTop: '0.5rem', padding: '0.45rem 0.6rem', borderRadius: '0.4rem', background: 'var(--color-code-bg)' }}>
          <span style={{ color: 'var(--color-accent)', fontWeight: 700 }}>{trace.codeword.slice(0, cur.s[5])}</span>
          <span style={{ color: 'var(--color-disabled-text)' }}>{trace.codeword.slice(cur.s[5])}</span>
        </div>
        <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
          Rows show the last {WINDOW} symbols. Hollow slices are the likeliest alternatives, placed where they sit in the
          interval; shaded rows are the speller at work on an unknown word. The whole {test === 'held' ? 'held-out' : 'EDGAR'} text
          under interpolated KN is one codeword of {fmt(code.length)} bits ({(code.length / code.chars).toFixed(4)} bits per character).
        </p>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Pick a unit — the same eight models, ranked by {METRICS.find((m) => m.key === metric)!.label}
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.4rem' }}>
          {TESTS.map((t) => (
            <Pill key={t.key} active={t.key === test} onClick={() => pickTest(t.key)} data={{ 'data-test': t.key }}>{t.label}</Pill>
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          {METRICS.map((m) => (
            <Pill key={m.key} active={m.key === metric} onClick={() => setMetric(m.key)} data={{ 'data-metric': m.key }}>{m.label}</Pill>
          ))}
        </div>
        <svg viewBox={`0 0 ${BW} ${BH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="units"
          data-order={unitRows.map((r) => r.model).join(',')}
          aria-label="Eight models ranked by the chosen unit; lower is better. The closed-vocabulary row is hatched because it is not a code.">
          <defs>
            <pattern id="pcl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" style={{ stroke: 'var(--color-accent)', strokeWidth: 2 }} />
            </pattern>
          </defs>
          {unitRows.map((r, i) => {
            const v = metricOf(r, metric);
            const y = 4 + i * BROW;
            return (
              <g key={r.model} data-unit-row={r.model}>
                <text x={BLABEL - 8} y={y + BROW / 2 + 3} textAnchor="end"
                  style={{ fontFamily: sans, fontSize: '11.5px', fill: r.is_code ? 'var(--color-text)' : 'var(--color-text-secondary)', fontStyle: r.is_code ? 'normal' : 'italic' }}>
                  {`#${i + 1}  ${MODEL_LABEL[r.model] ?? r.model}`}
                </text>
                <rect x={BLABEL} y={y + 3} width={Math.max(1, bx(v))} height={BROW - 8} rx={3}
                  style={{
                    fill: r.is_code ? (r.model === 'word' ? 'var(--color-accent)' : 'var(--color-accent-secondary)') : 'url(#pcl-hatch)',
                    fillOpacity: r.is_code ? 0.85 : 1, stroke: 'var(--color-accent)', strokeDasharray: r.is_code ? undefined : '4 3',
                  }} />
                <text x={BLABEL + Math.max(1, bx(v)) + 6} y={y + BROW / 2 + 3}
                  style={{ fontFamily: mono, fontSize: '11px', fill: 'var(--color-text-secondary)' }}>
                  {metricText(v, metric)}{r.is_code ? '' : ' (not a code)'}
                </text>
              </g>
            );
          })}
        </svg>
        <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.25rem 0 0.75rem' }}>
          Lower is better{metric === 'char' ? '' : ' (log scale)'}. Per-token perplexity divides by each model's own token
          count; bits per character and perplexity per word token divide every model by the same count, so they always
          agree. The open word model's bits per character is its arithmetic code: {fmt(code.length)} bits
          for {fmt(code.chars)} characters.
        </p>

        <label style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.75rem', fontFamily: sans, fontSize: '0.85rem', marginBottom: '0.25rem' }}>
          <span>words seen ≤ <strong data-readout="cutoff">{kLabel(cr)}</strong> times become ⟨unk⟩</span>
          <input type="range" min={0} max={cut.length - 1} step={1} value={cutIdx} aria-label="Unknown-word cutoff"
            onChange={(e) => setCutIdx(parseInt(e.target.value, 10))} style={{ flex: '1 1 8rem', minWidth: 0, accentColor: 'var(--color-accent)' }} />
        </label>
        <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="cutoff"
          aria-label="As more words become unknown, the closed-vocabulary perplexity falls while the honest code length in bits per character changes">
          <g transform={`translate(${CM.left},${CM.top})`}>
            <rect x={cx(cutIdx)! - 18} y={0} width={36} height={CH - CM.top - CM.bottom} style={{ fill: 'var(--color-muted-bg)' }} />
            <path d={lineL(cut)!} style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} data-series="closed-pp" />
            <path d={lineR(cut)!} style={{ fill: 'none', stroke: 'var(--color-accent-secondary)', strokeWidth: 2, strokeDasharray: '6 4' }} data-series="open-bpc" />
            {cut.map((r, i) => (
              <g key={i}>
                <circle cx={cx(i)} cy={cyL(r.closed_pp)} r={i === cutIdx ? 5 : 3.5} style={{ fill: 'var(--color-accent)' }} />
                <circle cx={cx(i)} cy={cyR(r.open_bpc)} r={i === cutIdx ? 5 : 3.5}
                  style={{ fill: 'var(--color-surface)', stroke: 'var(--color-accent-secondary)', strokeWidth: 2 }} />
                <text x={cx(i)} y={CH - CM.top - CM.bottom + 16} textAnchor="middle"
                  style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' }}>{`k = ${kLabel(r)}`}</text>
              </g>
            ))}
            {ticksL.map((t) => (
              <text key={`l${t}`} x={-8} y={cyL(t) + 3} textAnchor="end" style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-accent)' }}>
                {d3.format('~s')(t)}
              </text>
            ))}
            {cyR.ticks(4).map((t) => (
              <text key={`r${t}`} x={CW - CM.left - CM.right + 8} y={cyR(t) + 3} style={{ fontFamily: sans, fontSize: '10px', fill: 'var(--color-accent-secondary)' }}>
                {t.toFixed(1)}
              </text>
            ))}
          </g>
          <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>closed perplexity per token (log)</text>
          <text x={CW - 8} y={12} textAnchor="end" style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent-secondary)' }}>open bits per character</text>
        </svg>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.25rem' }}>
          <Readout id="vocab" label="vocabulary |V_k|" value={fmt(cr.V)} />
          <Readout id="oov" label="tokens sent to ⟨unk⟩" value={`${(cr.oov_share * 100).toFixed(2)}%`} />
          <Readout id="closed-pp" label="closed perplexity per token" value={cr.closed_pp >= 100 ? fmt(Math.round(cr.closed_pp)) : cr.closed_pp.toFixed(2)} emphasis />
          <Readout id="open-bpc" label="honest code, bits per character" value={cr.open_bpc.toFixed(4)} />
        </div>
      </figure>
      <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
        A perplexity above |V| = {fmt(V_SIZE)} is worse than a uniform guess over the vocabulary. The codeword lengths are
        exact (big-integer arithmetic coding); "closed" scores count every unknown word as one event and are not code lengths.
      </p>
    </div>
  );
}
