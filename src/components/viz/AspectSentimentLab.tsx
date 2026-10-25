import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/aspect-based-sentiment/aspect_based_sentiment.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_prop1_aspect_blind_ceiling, test_thm1b_direction_words_cannot_vote_their_meaning,
// test_thm1c_negation_is_not_representable, test_the_ladder, test_thm2_on_the_trained_model,
// test_prop2_aggregation_reaches_the_ceiling, test_edgar_readings, test_viz_payload). Change the .py, regenerate,
// never the reverse.
import payload from './data/aspect-sentiment-lab.json';

/**
 * Aspect Sentiment Laboratory — four figures for the aspect-based-sentiment topic.
 *   A. "Read a sentence": seven test aspects, word by word: each attention layer's weight on a word and its vote for
 *      positive over negative (Theorem 1(a)), with every reader's prediction against gold.
 *   B. "What one layer can read": accuracy by construction for each reader, the two contrast sets (Theorem 1(b)–(c)),
 *      and the shared layer's direction-word votes.
 *   C. "Attention isn't the explanation": reweighting the aspect-term words by r = 1 … 10⁶ moves the attention and
 *      keeps the decision and the vote shares (Theorem 2).
 *   D. "From aspects to tone": document tone from aspect polarities against the document-level readers and the Bayes
 *      ceiling (Proposition 2), and the EDGAR readings.
 */

type Rung = { att: number[]; vote: number[]; pred: number; p: number[]; margin: number };
type ReadRow = {
  tokens: string[]; span: [number, number]; category: string; valence: number; gold: number; construction: string;
  contrast: boolean; lexicon: number; blind: number; shared: Rung; aspect: Rung; bigram: Rung;
};
const READ = payload.read as unknown as { rows: ReadRow[] };
type RungStats = {
  id: string; all: number; contrast: number; errors: number; by: number[]; valence_quads: number; negation_quads: number;
};
const CON = payload.constructions as unknown as {
  constructions: string[]; n: number[]; rungs: RungStats[]; n_valence_quads: number; n_negation_quads: number;
  bound: number; direction_votes: Record<string, number>;
};
type Step = { log10_r: number; att: number[]; share: number[]; pred: number; term_mass: number; p: number[]; scale: number };
const IDF = payload.identifiability as unknown as {
  tokens: string[]; span: [number, number]; category: string; gold: number; term_words: boolean[]; steps: Step[];
};
const TONE = payload.tone as unknown as {
  bars: { label: string; acc: number }[]; ceiling: number; se: number; mean_k: number; errors: number; sentences: number;
};
const EDGAR = payload.edgar as unknown as {
  company: string; term: string; category: string; sentence: string; pred: string; p: number[]; lexicon: string;
}[];

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3);
const LABELS = ['negative', 'neutral', 'positive'];
const POL_COLOR = ['var(--color-badge-red-text)', 'var(--color-badge-blue-text)', 'var(--color-badge-green-text)'];
const READERS: { id: string; label: string }[] = [
  { id: 'lexicon', label: 'lexicon (Topic 18)' },
  { id: 'blind', label: 'aspect-blind LR' },
  { id: 'shared', label: 'shared votes' },
  { id: 'aspect', label: 'aspect values' },
  { id: 'bigram', label: 'aspect values + bigrams' },
];
const LAYERS = READERS.slice(2) as { id: 'shared' | 'aspect' | 'bigram'; label: string }[];

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

function Readout({ label, value, id, emphasis, color }: { label: string; value: string; id: string; emphasis?: boolean; color?: string }) {
  return (
    <div style={{ fontFamily: sans, fontSize: '0.8rem', minWidth: 0 }}>
      <div style={{ color: 'var(--color-text-secondary)' }}>{label}</div>
      <div data-readout={id} style={{ fontSize: '1.05rem', fontWeight: 600, color: color ?? (emphasis ? 'var(--color-accent)' : 'var(--color-text)'), overflowWrap: 'anywhere' }}>
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

function Key({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{ display: 'inline-block', width: '0.8rem', height: '0.5rem', background: s.color }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** Word chips: a bar above each word for its attention (relative to the sentence's largest), a bar below for its
 *  vote for positive over negative (green up-votes, red down-votes; height relative to the largest |vote|). */
function TokenStrip({ tokens, span, att, vote, mark }: {
  tokens: string[]; span: [number, number]; att: number[]; vote: number[]; mark?: boolean[];
}) {
  const amax = Math.max(...att, 1e-9);
  const vmax = Math.max(...vote.map(Math.abs), 1e-9);
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.25rem', margin: '0.4rem 0' }} data-panel="tokens">
      {tokens.map((t, i) => {
        const inSpan = i >= span[0] && i <= span[1];
        const v = vote[i];
        return (
          <div key={i} data-token={t} data-att={att[i]} data-vote={v}
            style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'stretch', minWidth: 0 }}>
            <div aria-hidden="true" style={{ height: '6px', background: 'var(--color-accent)', opacity: att[i] / amax, borderRadius: '2px' }} />
            <span style={{
              fontFamily: mono, fontSize: '0.78rem', padding: '0.1rem 0.25rem', textAlign: 'center',
              fontWeight: inSpan ? 700 : 400, textDecoration: inSpan ? 'underline' : undefined,
              background: mark?.[i] ? 'var(--color-muted-bg)' : undefined, color: 'var(--color-text)',
            }}>{t}</span>
            <div aria-hidden="true" style={{ height: '14px', display: 'flex', alignItems: 'flex-start', justifyContent: 'center' }}>
              <div style={{
                width: '70%', height: `${Math.max(1, (14 * Math.abs(v)) / vmax)}px`,
                background: v > 0 ? 'var(--color-badge-green-text)' : v < 0 ? 'var(--color-badge-red-text)' : 'var(--color-border)',
                opacity: Math.abs(v) / vmax < 0.01 ? 0.25 : 1,
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ----- A. Read a sentence -----

const ROW_LABELS = ['contrast: first aspect', 'contrast: second aspect', 'bare direction', 'cost-like verb', 'negated adjective', 'noun phrase', 'neutral'];

function ReadView() {
  const [ri, setRi] = useState(0);
  const [layer, setLayer] = useState<'shared' | 'aspect' | 'bigram'>('aspect');
  const r = READ.rows[ri];
  const L = r[layer];
  const preds: Record<string, number> = { lexicon: r.lexicon, blind: r.blind, shared: r.shared.pred, aspect: r.aspect.pred, bigram: r.bigram.pred };
  return (
    <figure style={{ margin: 0 }} data-view="read">
      <Caption>One sentence, one aspect: attention above each word, its vote below</Caption>
      <Row>
        {ROW_LABELS.map((lab, i) => (
          <Pill key={lab} active={ri === i} onClick={() => setRi(i)} data={{ 'data-row': String(i) }}>{lab}</Pill>
        ))}
      </Row>
      <Row>
        {LAYERS.map((l) => (
          <Pill key={l.id} active={layer === l.id} onClick={() => setLayer(l.id)} data={{ 'data-layer': l.id }}>{l.label}</Pill>
        ))}
      </Row>
      <div style={{ fontFamily: sans, fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>
        aspect <strong style={{ color: 'var(--color-text)' }}>{r.tokens.slice(r.span[0], r.span[1] + 1).join(' ')}</strong>
        {' '}· category {r.category} · valence {r.valence > 0 ? '+1 (up is good)' : '−1 (up is bad)'} · {r.construction}
      </div>
      <TokenStrip tokens={r.tokens} span={r.span} att={L.att} vote={L.vote} />
      <Key items={[{ label: 'attention', color: 'var(--color-accent)' }, { label: 'votes positive', color: 'var(--color-badge-green-text)' },
        { label: 'votes negative', color: 'var(--color-badge-red-text)' }]} />
      <Readouts>
        <Readout label="gold" value={LABELS[r.gold]} id="read-gold" color={POL_COLOR[r.gold]} />
        <Readout label={`${LAYERS.find((l) => l.id === layer)!.label}: p(${LABELS[L.pred]})`} value={f3(L.p[L.pred])} id="read-p" />
        <Readout label="logit gap, positive − negative" value={f3(L.margin)} id="read-margin" />
      </Readouts>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 1rem', fontFamily: sans, fontSize: '0.78rem', marginTop: '0.6rem' }} data-panel="readers">
        {READERS.map((rd) => {
          const ok = preds[rd.id] === r.gold;
          return (
            <span key={rd.id} data-reader={rd.id} data-ok={String(ok)} style={{ color: ok ? 'var(--color-text)' : 'var(--color-badge-red-text)' }}>
              {ok ? '✓' : '✗'} {rd.label}: {LABELS[preds[rd.id]]}
            </span>
          );
        })}
      </div>
      <Note>
        The vote bars sum to the logit gap. Attention only stretches them: switch layers on the bare direction or the
        negated adjective to see which words a vote can and cannot turn around.
      </Note>
    </figure>
  );
}

// ----- B. What one layer can read -----

const W = 640;
const BAR_ROWS = [...CON.constructions.map((c, i) => ({ key: c, label: c, n: CON.n[i] })), { key: 'contrast', label: 'contrast aspects', n: 0 }, { key: 'all', label: 'all aspects', n: 0 }];

function LadderView() {
  const [reader, setReader] = useState('shared');
  const s = CON.rungs.find((r) => r.id === reader)!;
  const vals = [...s.by, s.contrast, s.all];
  const LW = 150, IW = W - LW - 20, RH = 24;
  const x = d3.scaleLinear().domain([0, 1]).range([0, IW]);
  const words = Object.entries(CON.direction_votes);
  const vx = d3.scaleLinear().domain([-8, 8]).range([0, 300]);
  return (
    <figure style={{ margin: 0 }} data-view="ladder">
      <Caption>Accuracy by construction, for each reader</Caption>
      <Row>
        {READERS.map((rd) => (
          <Pill key={rd.id} active={reader === rd.id} onClick={() => setReader(rd.id)} data={{ 'data-reader': rd.id }}>{rd.label}</Pill>
        ))}
      </Row>
      <svg viewBox={`0 0 ${W} ${BAR_ROWS.length * RH + 30}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="constructions"
        aria-label={`${reader}: accuracy by construction; all aspects ${f3(s.all)}, contrast aspects ${f3(s.contrast)}`}>
        <text x={8} y={14} style={titleText}>share of test aspects read correctly</text>
        <g transform={`translate(${LW},24)`}>
          {BAR_ROWS.map((b, i) => {
            const a = vals[i];
            return (
              <g key={b.key} transform={`translate(0,${i * RH})`} data-row={b.key}>
                <text x={-8} y={14} textAnchor="end" style={{ ...axisText, fontSize: '10.5px', fill: 'var(--color-text)' }}>{b.label}</text>
                <rect x={0} y={3} width={IW} height={16} style={{ fill: 'var(--color-muted-bg)' }} />
                <rect x={0} y={3} width={x(a)} height={16} style={{ fill: a < 0.6 ? 'var(--color-badge-red-text)' : 'var(--color-accent)' }} />
                <text x={a > 0.5 ? x(a) - 5 : x(a) + 5} y={15} textAnchor={a > 0.5 ? 'end' : 'start'}
                  style={{ ...axisText, fill: a > 0.5 ? 'var(--color-bg)' : 'var(--color-text)' }}>
                  {f3(a)}{b.n ? ` · n = ${b.n}` : ''}
                </text>
              </g>
            );
          })}
          <line x1={x(CON.bound)} x2={x(CON.bound)} y1={(BAR_ROWS.length - 1) * RH} y2={BAR_ROWS.length * RH}
            style={{ stroke: 'var(--color-text)', strokeWidth: 1.5, strokeDasharray: '3 2' }} />
        </g>
      </svg>
      <Note>Dashed: the aspect-blind ceiling {f3(CON.bound)} on all aspects (Proposition 1).</Note>
      <Readouts>
        <Readout label="valence quadruples all right" value={`${f3(s.valence_quads)} of ${CON.n_valence_quads}`} id="ladder-valence" emphasis />
        <Readout label="negation quadruples all right" value={`${f3(s.negation_quads)} of ${CON.n_negation_quads}`} id="ladder-negation" emphasis />
        <Readout label="test aspects misread" value={String(s.errors)} id="ladder-errors" />
      </Readouts>
      <svg viewBox="0 0 400 190" style={{ width: '100%', maxWidth: '26rem', height: 'auto', marginTop: '0.9rem' }} role="img" data-panel="votes"
        aria-label={`shared layer votes for positive over negative: ${words.map(([w, v]) => `${w} ${f3(v)}`).join(', ')}`}>
        <text x={4} y={12} style={titleText}>shared-vote layer: each word's vote for positive over negative</text>
        <g transform="translate(80,24)">
          <line x1={vx(0)} x2={vx(0)} y1={0} y2={words.length * 19} style={{ stroke: 'var(--color-border)' }} />
          {words.map(([w, v], i) => (
            <g key={w} transform={`translate(0,${i * 19})`} data-word={w}>
              <text x={-6} y={12} textAnchor="end" style={{ ...axisText, fill: 'var(--color-text)' }}>{w}</text>
              <rect x={Math.min(vx(0), vx(v))} y={3} width={Math.abs(vx(v) - vx(0))} height={12}
                style={{ fill: v > 0 ? 'var(--color-badge-green-text)' : 'var(--color-badge-red-text)' }} />
              <text x={v > 0 ? vx(v) + 4 : vx(v) - 4} y={13} textAnchor={v > 0 ? 'start' : 'end'} style={axisText}>{f3(v)}</text>
            </g>
          ))}
        </g>
      </svg>
      <Note>
        With shared votes a word votes the same way for every aspect, so reading "revenue rose" and "costs rose" needs
        both direction words on one side (Theorem 1(b)): rose, fell, declined, and decreased are; increased and climbed
        are not, and every quadruple that pairs them with a fall word is lost. No single-token layer reads a negation
        quadruple (Theorem 1(c)); a bigram item, or the lexicon's hand-written scope rule, does.
      </Note>
    </figure>
  );
}

// ----- C. Attention isn't the explanation -----

function IdentifiabilityView() {
  const [k, setK] = useState(0);
  const st = IDF.steps[k];
  const rTxt = st.log10_r === 0 ? '1' : `10^${st.log10_r}`;
  return (
    <figure style={{ margin: 0 }} data-view="identifiability">
      <Caption>Reweighting the aspect words: the attention moves, the decision does not</Caption>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.4rem 0.75rem', fontFamily: sans, fontSize: '0.8rem', marginBottom: '0.45rem' }}>
        <label htmlFor="absa-lab-r" style={{ minWidth: 0 }}>weight r on aspect-term words</label>
        <input id="absa-lab-r" type="range" min={0} max={IDF.steps.length - 1} step={1} value={k} data-control="r"
          onChange={(e) => setK(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={`r = ${rTxt}`} />
        <span data-readout="r-value" style={{ fontFamily: mono, minWidth: 0 }}>
          r = {st.log10_r === 0 ? '1' : <>10<sup>{st.log10_r}</sup></>}
        </span>
      </div>
      <div style={{ fontFamily: sans, fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>
        aspect <strong style={{ color: 'var(--color-text)' }}>{IDF.tokens.slice(IDF.span[0], IDF.span[1] + 1).join(' ')}</strong>
        {' '}· gold {LABELS[IDF.gold]} · shaded words are aspect-term words · bars below are vote shares
      </div>
      <TokenStrip tokens={IDF.tokens} span={IDF.span} att={st.att} vote={st.share} mark={IDF.term_words} />
      <Readouts>
        <Readout label="decision" value={LABELS[st.pred]} id="idf-decision" color={POL_COLOR[st.pred]} />
        <Readout label="attention on aspect-term words" value={f3(st.term_mass)} id="idf-mass" emphasis />
        <Readout label={`p(${LABELS[st.pred]})`} value={f3(st.p[st.pred])} id="idf-p" />
        <Readout label="logit scale λ" value={st.scale.toFixed(4)} id="idf-scale" />
      </Readouts>
      <Note>
        Theorem 2: adding log r to the aspect words' scores and dividing their values by r multiplies every logit by
        one positive number λ. The decision and the vote shares stay; the attention can be put anywhere; the
        probabilities flatten, because λ acts as a temperature.
      </Note>
    </figure>
  );
}

// ----- D. From aspects to tone -----

function ToneView() {
  const LW = 200, IW = W - LW - 20, RH = 26;
  const x = d3.scaleLinear().domain([0.5, 1]).range([0, IW]);
  const n = TONE.bars.length;
  return (
    <figure style={{ margin: 0 }} data-view="tone">
      <Caption>Document tone, read whole or aspect by aspect</Caption>
      <svg viewBox={`0 0 ${W} ${n * RH + 50}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="tone"
        aria-label={`document accuracy: ${TONE.bars.map((b) => `${b.label} ${f3(b.acc)}`).join(', ')}; Bayes ceiling ${f3(TONE.ceiling)}`}>
        <text x={8} y={14} style={titleText}>three-class tone accuracy on 1,500 test filings (axis from 0.5)</text>
        <g transform={`translate(${LW},26)`}>
          <rect x={x(TONE.ceiling - 2 * TONE.se)} y={0} width={x(TONE.ceiling + 2 * TONE.se) - x(TONE.ceiling - 2 * TONE.se)} height={n * RH}
            style={{ fill: 'var(--color-muted-bg)' }} />
          {TONE.bars.map((b, i) => (
            <g key={b.label} transform={`translate(0,${i * RH})`} data-bar={b.label}>
              <text x={-8} y={15} textAnchor="end" style={{ ...axisText, fontSize: '10.5px', fill: 'var(--color-text)' }}>{b.label}</text>
              <rect x={0} y={4} width={x(b.acc)} height={16} style={{ fill: i >= n - 2 ? 'var(--color-accent)' : 'var(--color-text-secondary)' }} />
              <text x={x(b.acc) - 5} y={16} textAnchor="end" style={{ ...axisText, fill: 'var(--color-bg)' }}>{f3(b.acc)}</text>
            </g>
          ))}
          <line x1={x(TONE.ceiling)} x2={x(TONE.ceiling)} y1={-4} y2={n * RH} style={{ stroke: 'var(--color-text)', strokeWidth: 1.5, strokeDasharray: '3 2' }} />
          <text x={x(TONE.ceiling)} y={n * RH + 14} textAnchor="middle" style={axisText}>Bayes ceiling {f3(TONE.ceiling)} (shaded ± 2 SE)</text>
        </g>
      </svg>
      <Readouts>
        <Readout label="tone-bearing sentences per filing" value={f3(TONE.mean_k)} id="tone-k" />
        <Readout label="sentences the bigram layer misreads" value={`${TONE.errors} of ${TONE.sentences}`} id="tone-errors" emphasis />
        <Readout label="standard error at n = 1,500" value={f3(TONE.se)} id="tone-se" />
      </Readouts>
      <div style={{ marginTop: '1rem' }} data-panel="edgar">
        <div style={{ fontFamily: sans, fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.4rem' }}>
          Aspect mentions in three 10-K excerpts (no gold: readings, not scores)
        </div>
        {EDGAR.map((e, i) => (
          <div key={i} data-edgar={i} style={{ borderTop: '1px solid var(--color-border)', padding: '0.4rem 0', fontFamily: sans, fontSize: '0.78rem' }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.2rem 0.9rem' }}>
              <span style={{ fontWeight: 600 }}>{e.company}</span>
              <span>aspect <em>{e.term}</em> ({e.category})</span>
              <span>bigram layer: <span style={{ color: POL_COLOR[LABELS.indexOf(e.pred)], fontWeight: 600 }}>{e.pred}</span>{' '}
                p = {f3(Math.max(...e.p))}</span>
              <span>lexicon: {e.lexicon}</span>
            </div>
            <div style={{ color: 'var(--color-text-secondary)', marginTop: '0.2rem', overflowWrap: 'anywhere' }}>{e.sentence}</div>
          </div>
        ))}
      </div>
      <Note>
        The filings' tone lives in 5 to 8 aspect sentences each; counting their polarities is Proposition 2's Bayes rule.
        On real MD&amp;A the layer, which learned "neutral" from a few templates, calls descriptive mentions polar at
        about even odds.
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'read', label: 'Read a sentence' },
  { key: 'ladder', label: 'What one layer can read' },
  { key: 'identifiability', label: "Attention isn't the explanation" },
  { key: 'tone', label: 'From aspects to tone' },
] as const;

export default function AspectSentimentLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('read');
  return (
    <div data-lab="aspect-sentiment" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'read' && <ReadView />}
      {view === 'ladder' && <LadderView />}
      {view === 'identifiability' && <IdentifiabilityView />}
      {view === 'tone' && <ToneView />}
    </div>
  );
}
