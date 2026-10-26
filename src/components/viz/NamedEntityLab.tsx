import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/named-entity-recognition/named_entity_recognition.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_prop1_encodings_are_bijections, test_prop2_strict_for_token_local_features,
// test_the_word_crf_memorizes_names, test_dropout_recipes, test_encoder_context, test_case_is_most_of_the_signal,
// test_crf_head_vs_softmax_head, test_bioes_buys_nothing_here, test_thm3_on_the_finance_taggers,
// test_prop3_on_the_finance_taggers). Change the .py, regenerate, never the reverse.
import payload from './data/named-entity-lab.json';

/**
 * Named Entity Laboratory — five figures for the named-entity-recognition topic.
 *   A. "Encodings": the example sentence in BIO and BIOES, the one-type validity masks, valid vs all sequences, and
 *      Proposition 2's two-sentence counterexample.
 *   B. "Held-out names": every tagger on the seen and unseen tests, two encoder readings, four match criteria,
 *      per-type F1, and the example sentence's predicted spans.
 *   C. "Dropout": substitution vs deletion across dropout rates (Proposition 5's shortcut).
 *   D. "Heads & constraints": softmax vs CRF heads on frozen encoder states, raw vs constrained (Theorem 1), and BIOES.
 *   E. "F1 decoding": threshold sweeps over span marginals against half of each model's own F-star (Theorem 3).
 */

type Span = [number, number, string];
type Reading = {
  f1: number; p: number; r: number; raw_f1: number; invalid: number;
  schemes: Record<'strict' | 'boundary' | 'type' | 'partial', number>;
  types: Record<string, number>; exposure: Record<'seen' | 'unseen', number>; example: Span[] | null;
};
type ModelEntry = { feats: string; dropout: string; p: number; scheme: string; head: string; readings: Record<string, Reading> };
const ENC = payload.encodings as unknown as {
  tokens: string[]; spans: Span[]; tags: Record<'bio' | 'bioes', string[]>;
  masks: Record<'bio' | 'bioes', { tags: string[]; start: number[]; trans: number[][]; end: number[] }>;
  counts: { T: number; sets: number; bio: number; bioes: number }[];
  prop2: { data: [string[], Span[]][]; bio: number[]; bioes: number[]; 'bio+left': number[] };
};
const MODELS = payload.models as unknown as Record<string, ModelEntry>;
const HEADLINE = payload.headline as string;
type Curve = { f1: number[]; p: number[]; r: number[] };
const SW = payload.sweep as unknown as { p: number[]; substitute: Record<'seen' | 'unseen', Curve>; delete: Record<'seen' | 'unseen', Curve> };
type Sweep = Curve & { tau: number[]; expected: number[]; F_star: number; viterbi_f1: number; f1_decoded: number; example: [number, number, string, number][] | null };
const TH = payload.thresholds as unknown as Record<string, { unk: boolean; seen: Sweep; unseen: Sweep }>;
const EXPO = payload.exposure as { name_words: number; in_vocab: number };
const TYPES = ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'SECTOR', 'DATE', 'MONEY', 'PERCENT'];

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const f3 = (v: number) => v.toFixed(3);
const W = 640;

const TYPE_COLORS: Record<string, string> = {
  ORG: 'var(--color-accent)', PERSON: 'var(--color-badge-blue-text)', GPE: 'var(--color-badge-green-text)',
  PRODUCT: 'var(--color-badge-amber-text)', SECTOR: 'var(--color-accent-secondary)', DATE: 'var(--color-badge-red-text)',
  MONEY: 'var(--color-text-secondary)', PERCENT: 'var(--color-text)',
};

function Pill({ active, onClick, children, data, disabled }: { active: boolean; onClick: () => void; children: ReactNode; data: Record<string, string>; disabled?: boolean }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} disabled={disabled} {...data}
      style={{
        fontFamily: sans, fontSize: '0.75rem', padding: '0.2rem 0.55rem', borderRadius: '999px',
        cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.45 : 1,
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

function Key({ items }: { items: { label: string; color: string; dash?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)', marginTop: '0.4rem' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <span aria-hidden="true" style={{
            display: 'inline-block', width: '0.9rem', height: s.dash ? 0 : '0.2rem',
            borderTop: s.dash ? `2px dashed ${s.color}` : undefined, background: s.dash ? undefined : s.color,
          }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** The sentence's tokens with spans underlined in their type's color. */
function SpanLine({ tokens, spans, id, faint }: { tokens: string[]; spans: Span[]; id: string; faint?: boolean[] }) {
  const owner = tokens.map((_, t) => spans.findIndex(([s, e]) => s <= t && t < e));
  return (
    <div data-spans={id} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.15rem 0.35rem', fontFamily: mono, fontSize: '0.8rem' }}>
      {tokens.map((w, t) => {
        const j = owner[t];
        const sp = j >= 0 ? spans[j] : null;
        return (
          <span key={t} style={{
            padding: '0.05rem 0.2rem', borderBottom: sp ? `3px solid ${TYPE_COLORS[sp[2]]}` : '3px solid transparent',
            opacity: faint && j >= 0 && faint[j] ? 0.45 : 1,
          }}>
            {w}{sp && t === sp[1] - 1 ? <sub style={{ fontFamily: sans, fontSize: '0.6rem', color: TYPE_COLORS[sp[2]] }}> {sp[2]}</sub> : null}
          </span>
        );
      })}
      {spans.length === 0 && <span style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>(no spans)</span>}
    </div>
  );
}

// ----- A. Encodings -----

function EncodingsView() {
  const [scheme, setScheme] = useState<'bio' | 'bioes'>('bio');
  const M = ENC.masks[scheme];
  const n = M.tags.length;
  const CW = 54, CH = 22, ML = 62, MT = 22;
  const WT = ML + n * CW + 8, HT = MT + (n + 2) * CH + 10;
  const P2 = ENC.prop2;
  const sums = (['bio', 'bioes', 'bio+left'] as const).map((k) => P2[k][0] + P2[k][1]);
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Tags encode span sets; validity is a first-order constraint</Caption>
      <Row>
        <Label>scheme</Label>
        <Pill active={scheme === 'bio'} onClick={() => setScheme('bio')} data={{ 'data-scheme': 'bio' }}>BIO</Pill>
        <Pill active={scheme === 'bioes'} onClick={() => setScheme('bioes')} data={{ 'data-scheme': 'bioes' }}>BIOES</Pill>
      </Row>
      <div data-panel="tags" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.5rem', fontFamily: mono, fontSize: '0.78rem', marginBottom: '0.6rem' }}>
        {ENC.tokens.map((w, t) => {
          const tag = ENC.tags[scheme][t];
          return (
            <span key={t} style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center' }}>
              <span>{w}</span>
              <span data-tag={t} style={{ fontSize: '0.68rem', color: tag === 'O' ? 'var(--color-text-secondary)' : TYPE_COLORS[tag.slice(2)] }}>{tag}</span>
            </span>
          );
        })}
      </div>
      <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
        <svg viewBox={`0 0 ${WT} ${HT}`} style={{ width: '100%', maxWidth: `${WT * 1.2}px`, height: 'auto' }} role="img" data-panel="masks"
          aria-label={`${scheme.toUpperCase()} validity masks for one type: allowed starts, transitions, and ends`}>
          {M.tags.map((t, k) => (
            <text key={`c${k}`} x={ML + k * CW + CW / 2} y={MT - 8} textAnchor="middle" style={{ ...axisText, fontFamily: mono }}>{t}</text>
          ))}
          {[...M.tags.map((t) => `${t} →`), 'start', 'end'].map((r, i) => (
            <text key={`r${i}`} x={ML - 6} y={MT + i * CH + CH / 2 + 3} textAnchor="end" style={{ ...axisText, fontFamily: mono }}>{r}</text>
          ))}
          {[...M.trans, M.start, M.end].map((row, i) => row.map((v, k) => (
            <rect key={`${i}-${k}`} x={ML + k * CW + 1} y={MT + i * CH + 1} width={CW - 2} height={CH - 2} data-mask={`${i}-${k}`}
              style={{ fill: v ? 'var(--color-accent)' : 'var(--color-surface)', fillOpacity: v ? 0.55 : 1, stroke: 'var(--color-border)', strokeWidth: 0.6 }} />
          )))}
        </svg>
      </div>
      <Key items={[{ label: 'allowed (filled) — the rest are set to −∞ in constrained decoding (Theorem 1)', color: 'var(--color-accent)' }]} />
      <div role="table" data-panel="counts" aria-label="valid span sets against all tag sequences"
        style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '2px', fontFamily: sans, fontSize: '0.72rem', marginTop: '0.7rem' }}>
        {['tokens T', 'span sets', 'BIO sequences', 'BIOES sequences'].map((h) => (
          <div role="columnheader" key={h} style={{ color: 'var(--color-text-secondary)', padding: '0.15rem' }}>{h}</div>
        ))}
        {ENC.counts.map((c) => [
          <div role="cell" key={`t${c.T}`} style={{ padding: '0.15rem' }}>{c.T}</div>,
          <div role="cell" key={`s${c.T}`} style={{ padding: '0.15rem', fontFamily: mono, overflowWrap: 'anywhere' }}>{c.sets.toLocaleString('en-US')}</div>,
          <div role="cell" key={`b${c.T}`} style={{ padding: '0.15rem', fontFamily: mono, overflowWrap: 'anywhere' }}>{c.bio.toLocaleString('en-US')}</div>,
          <div role="cell" key={`e${c.T}`} style={{ padding: '0.15rem', fontFamily: mono, overflowWrap: 'anywhere' }}>{c.bioes.toLocaleString('en-US')}</div>,
        ])}
      </div>
      <Note>Eight entity types. The valid sequences of either scheme are exactly the encodings of the span sets (Proposition 1).</Note>
      <div style={{ marginTop: '0.8rem', fontFamily: sans, fontSize: '0.78rem', fontWeight: 600 }}>
        Proposition 2: “a d” → one-token ORG, “c d” → two-token ORG
      </div>
      <Readouts>
        <Readout label="BIO, token-local: gold probabilities (sum)" value={`${f3(P2.bio[0])}, ${f3(P2.bio[1])} (${f3(sums[0])})`} id="p2-bio" />
        <Readout label="BIOES, token-local" value={`${f3(P2.bioes[0])}, ${f3(P2.bioes[1])}`} id="p2-bioes" emphasis />
        <Readout label="BIO with the left word" value={`${f3(P2['bio+left'][0])}, ${f3(P2['bio+left'][1])}`} id="p2-left" />
      </Readouts>
      <Note>
        Given B on the first token, a token-local BIO chain decides the second tag from “d” alone, so the two gold sets
        share at most probability 1. BIOES separates them by the first tag (S or B); one word of left context does too.
      </Note>
    </figure>
  );
}

// ----- B. Held-out names -----

const SCHEMES = ['strict', 'boundary', 'type', 'partial'] as const;

function HeldOutView() {
  const [name, setName] = useState('words');
  const [test, setTest] = useState<'seen' | 'unseen'>('unseen');
  const [unk, setUnk] = useState(false);
  const [match, setMatch] = useState<(typeof SCHEMES)[number]>('strict');
  const m = MODELS[name];
  const hasEnc = m.feats === 'encoder' || m.feats === 'both';
  const key = `${test}${hasEnc && unk ? '-unk' : ''}`;
  const r = m.readings[key];
  const ex = (m.readings[`unseen${hasEnc && unk ? '-unk' : ''}`] as Reading).example ?? [];
  const H = 200, MG = { l: 70, r: 40, t: 6, b: 18 };
  const IW = W - MG.l - MG.r;
  const x = d3.scaleLinear().domain([0, 1]).range([0, IW]);
  const bh = (H - MG.t - MG.b) / TYPES.length;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Who finds a name the tagger never saw</Caption>
      <Row>
        <Label>tagger</Label>
        {Object.keys(MODELS).map((k) => (
          <Pill key={k} active={name === k} onClick={() => setName(k)} data={{ 'data-model': k }}>{k === HEADLINE ? `${k} (headline)` : k}</Pill>
        ))}
      </Row>
      <Row>
        <Label>test</Label>
        <Pill active={test === 'seen'} onClick={() => setTest('seen')} data={{ 'data-test': 'seen' }}>seen roster</Pill>
        <Pill active={test === 'unseen'} onClick={() => setTest('unseen')} data={{ 'data-test': 'unseen' }}>held-out roster</Pill>
        <Label>encoder reads unseen words</Label>
        <Pill active={!unk} onClick={() => setUnk(false)} data={{ 'data-reading': 'ids' }} disabled={!hasEnc}>by id</Pill>
        <Pill active={unk} onClick={() => setUnk(true)} data={{ 'data-reading': 'unk' }} disabled={!hasEnc}>as [UNK]</Pill>
      </Row>
      <Row>
        <Label>match</Label>
        {SCHEMES.map((s) => (
          <Pill key={s} active={match === s} onClick={() => setMatch(s)} data={{ 'data-match': s }}>{s}</Pill>
        ))}
      </Row>
      <Readouts>
        <Readout label={`span F1 (${match})`} value={f3(r.schemes[match])} id="f1" emphasis />
        <Readout label="strict precision / recall" value={`${f3(r.p)} / ${f3(r.r)}`} id="pr" />
        <Readout label="strict F1, seen / unseen mentions" value={`${f3(r.exposure.seen)} / ${f3(r.exposure.unseen)}`} id="exposure" />
      </Readouts>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', marginTop: '0.5rem' }} role="img" data-panel="types"
        aria-label={`strict F1 by type: ${TYPES.map((t) => `${t} ${f3(r.types[t])}`).join(', ')}`}>
        <g transform={`translate(${MG.l},${MG.t})`}>
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={0} y2={bh * TYPES.length} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={x(v)} y={bh * TYPES.length + 12} textAnchor="middle" style={axisText}>{v}</text>
            </g>
          ))}
          {TYPES.map((t, i) => (
            <g key={t}>
              <text x={-6} y={i * bh + bh / 2 + 3} textAnchor="end" style={axisText}>{t}</text>
              <rect x={0} y={i * bh + 3} width={x(r.types[t])} height={bh - 6} data-bar={t} style={{ fill: TYPE_COLORS[t], fillOpacity: 0.8 }} />
              <text x={x(r.types[t]) + 4} y={i * bh + bh / 2 + 3} style={axisText}>{f3(r.types[t])}</text>
            </g>
          ))}
        </g>
      </svg>
      <div style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.6rem 0 0.25rem' }}>held-out example — gold</div>
      <SpanLine tokens={ENC.tokens} spans={ENC.spans} id="gold" />
      <div style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0.25rem' }}>this tagger</div>
      <SpanLine tokens={ENC.tokens} spans={ex} id="pred" />
      <Note>
        Strict F1 per type (bars). The word CRF predicts no organization anywhere in the held-out test. Relax the match to
        see boundary errors: a tagger that finds a name but misplaces its edges gains under “type” and “partial”
        (Proposition 3). {EXPO.in_vocab} of the {EXPO.name_words} held-out name words were in the encoder&rsquo;s
        pretraining vocabulary.
      </Note>
    </figure>
  );
}

// ----- shared line chart -----

type Series = { label: string; ys: number[]; color: string; dash?: boolean; id: string };

function LineChart({ xs, series, xLabel, marks, hLines, cursor, yDomain, id }: {
  xs: number[]; series: Series[]; xLabel: string; marks?: { x: number; label: string }[];
  hLines?: { y: number; label: string; color: string }[]; cursor?: number; yDomain: [number, number]; id: string;
}) {
  const H = 250, M = { l: 40, r: 14, t: 12, b: 36 };
  const IW = W - M.l - M.r, IH = H - M.t - M.b;
  const x = d3.scaleLinear().domain([xs[0], xs[xs.length - 1]]).range([0, IW]);
  const y = d3.scaleLinear().domain(yDomain).range([IH, 0]);
  const line = (ys: number[]) => d3.line<number>().x((_, i) => x(xs[i])).y((v) => y(v))(ys) ?? '';
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id}
      aria-label={series.map((s) => `${s.label}: ${s.ys.map(f3).join(', ')}`).join('; ')}>
      <g transform={`translate(${M.l},${M.t})`}>
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line x1={0} x2={IW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-6} y={y(t) + 3} textAnchor="end" style={axisText}>{t}</text>
          </g>
        ))}
        {x.ticks(5).map((t) => (
          <text key={t} x={x(t)} y={IH + 14} textAnchor="middle" style={axisText}>{t}</text>
        ))}
        <text x={IW / 2} y={IH + 30} textAnchor="middle" style={axisText}>{xLabel}</text>
        {hLines?.map((h) => (
          <line key={h.label} x1={0} x2={IW} y1={y(h.y)} y2={y(h.y)} data-hline={h.label} style={{ stroke: h.color, strokeWidth: 1.2, strokeDasharray: '2 3' }} />
        ))}
        {marks?.map((mk) => (
          <g key={mk.label}>
            <line x1={x(mk.x)} x2={x(mk.x)} y1={0} y2={IH} data-mark={mk.label} style={{ stroke: 'var(--color-text)', strokeWidth: 1, strokeDasharray: '5 3' }} />
            <text x={x(mk.x) + 4} y={10} style={axisText}>{mk.label}</text>
          </g>
        ))}
        {cursor !== undefined && (
          <line x1={x(cursor)} x2={x(cursor)} y1={0} y2={IH} style={{ stroke: 'var(--color-accent)', strokeWidth: 1.5, opacity: 0.6 }} />
        )}
        {series.map((s) => (
          <path key={s.id} d={line(s.ys)} data-series={s.id}
            style={{ fill: 'none', stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash ? '5 3' : undefined }} />
        ))}
      </g>
    </svg>
  );
}

// ----- C. Dropout -----

function DropoutView() {
  const [i, setI] = useState(2);
  const [metric, setMetric] = useState<'f1' | 'p'>('f1');
  const p = SW.p[i];
  const series: Series[] = [
    { label: 'deletion, unseen test', ys: SW.delete.unseen[metric], color: 'var(--color-accent)', id: 'delete-unseen' },
    { label: 'deletion, seen test', ys: SW.delete.seen[metric], color: 'var(--color-accent)', dash: true, id: 'delete-seen' },
    { label: 'substitution, unseen test', ys: SW.substitute.unseen[metric], color: 'var(--color-badge-blue-text)', id: 'substitute-unseen' },
    { label: 'substitution, seen test', ys: SW.substitute.seen[metric], color: 'var(--color-badge-blue-text)', dash: true, id: 'substitute-seen' },
  ];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Name dropout: substituting &lt;unk&gt; teaches a shortcut; deleting the name&rsquo;s features does not</Caption>
      <Row>
        <Label>measure</Label>
        <Pill active={metric === 'f1'} onClick={() => setMetric('f1')} data={{ 'data-metric': 'f1' }}>strict F1</Pill>
        <Pill active={metric === 'p'} onClick={() => setMetric('p')} data={{ 'data-metric': 'p' }}>precision</Pill>
      </Row>
      <LineChart xs={SW.p} series={series} xLabel="dropout rate p (word-feature CRF)" cursor={p} yDomain={[0, 1]} id="sweep" />
      <Key items={series.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))} />
      <Row style={{ marginTop: '0.6rem' }}>
        <Label>p</Label>
        <input type="range" min={0} max={SW.p.length - 1} step={1} value={i} onChange={(e) => setI(Number(e.target.value))}
          aria-label="dropout rate" data-control="p" style={{ flex: '1 1 10rem', minWidth: 0 }} />
        <span style={{ fontFamily: mono, fontSize: '0.8rem' }} data-readout="p">p = {p}</span>
      </Row>
      <Readouts>
        <Readout label="deletion: seen / unseen" value={`${f3(SW.delete.seen[metric][i])} / ${f3(SW.delete.unseen[metric][i])}`} id="delete" emphasis />
        <Readout label="substitution: seen / unseen" value={`${f3(SW.substitute.seen[metric][i])} / ${f3(SW.substitute.unseen[metric][i])}`} id="substitute" />
      </Readouts>
      <Note>
        Substitution makes &lt;unk&gt; a feature that, in training, appears only on entities, so its weight for O is
        negative (Proposition 5) and every unseen word at test, name or not, leans toward an entity. Deletion adds no
        feature. At p = 1 even million, percent, and quarter never show their identity in training.
      </Note>
    </figure>
  );
}

// ----- D. Heads & constraints -----

function HeadsView() {
  const [test, setTest] = useState<'seen' | 'unseen'>('unseen');
  const soft = MODELS['softmax on encoder'].readings[test];
  const crf = MODELS.encoder.readings[test];
  const bio = MODELS['words, deletion dropout'].readings[test];
  const bioes = MODELS['words, deletion dropout, BIOES'].readings[test];
  const bars = [
    { label: 'softmax, raw', v: soft.raw_f1, id: 'soft-raw', color: 'var(--color-badge-blue-text)' },
    { label: 'softmax, constrained', v: soft.f1, id: 'soft-con', color: 'var(--color-badge-blue-text)' },
    { label: 'CRF, raw', v: crf.raw_f1, id: 'crf-raw', color: 'var(--color-accent)' },
    { label: 'CRF, constrained', v: crf.f1, id: 'crf-con', color: 'var(--color-accent)' },
  ];
  const H = 170, MG = { l: 130, r: 44, t: 6, b: 18 };
  const IW = W - MG.l - MG.r;
  const x = d3.scaleLinear().domain([0, 1]).range([0, IW]);
  const bh = (H - MG.t - MG.b) / bars.length;
  return (
    <figure style={{ margin: 0 }}>
      <Caption>On frozen encoder states, the CRF&rsquo;s transitions are worth more than the constraint</Caption>
      <Row>
        <Label>test</Label>
        <Pill active={test === 'seen'} onClick={() => setTest('seen')} data={{ 'data-test': 'seen' }}>seen roster</Pill>
        <Pill active={test === 'unseen'} onClick={() => setTest('unseen')} data={{ 'data-test': 'unseen' }}>held-out roster</Pill>
      </Row>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="heads"
        aria-label={bars.map((b) => `${b.label} ${f3(b.v)}`).join(', ')}>
        <g transform={`translate(${MG.l},${MG.t})`}>
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={x(v)} x2={x(v)} y1={0} y2={bh * bars.length} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
              <text x={x(v)} y={bh * bars.length + 12} textAnchor="middle" style={axisText}>{v}</text>
            </g>
          ))}
          {bars.map((b, i) => (
            <g key={b.id}>
              <text x={-6} y={i * bh + bh / 2 + 3} textAnchor="end" style={axisText}>{b.label}</text>
              <rect x={0} y={i * bh + 4} width={x(b.v)} height={bh - 8} data-bar={b.id}
                style={{ fill: b.color, fillOpacity: b.label.endsWith('raw') ? 0.45 : 0.85 }} />
              <text x={x(b.v) + 4} y={i * bh + bh / 2 + 3} style={axisText}>{f3(b.v)}</text>
            </g>
          ))}
        </g>
      </svg>
      <Readouts>
        <Readout label="invalid BIO decodes: softmax / CRF" value={`${f3(soft.invalid)} / ${f3(crf.invalid)}`} id="invalid" />
        <Readout label="constrained F1: softmax / CRF" value={`${f3(soft.f1)} / ${f3(crf.f1)}`} id="heads-f1" emphasis />
        <Readout label="word CRF, deletion: BIO / BIOES" value={`${f3(bio.f1)} / ${f3(bioes.f1)}`} id="bioes" />
      </Readouts>
      <Note>
        Strict span F1. The constraint (Theorem 1) repairs what the softmax head cannot represent, but the frozen states do
        not know where a name ends; the transitions do. BIOES adds nothing for features that see their neighbors
        (Proposition 2).
      </Note>
    </figure>
  );
}

// ----- E. F1 decoding -----

function F1View() {
  const names = Object.keys(TH);
  const [name, setName] = useState(names[0]);
  const [test, setTest] = useState<'seen' | 'unseen'>('unseen');
  const c = TH[name][test];
  const half = c.F_star / 2;
  const [k, setK] = useState(Math.round(half / 0.02) - 1);
  const tau = c.tau[k];
  const ex = TH[name].unseen.example ?? [];
  const chosen = ex.filter(([, , , q]) => q > tau);
  const series: Series[] = [
    { label: 'realized F1', ys: c.f1, color: 'var(--color-accent)', id: 'f1' },
    { label: 'precision', ys: c.p, color: 'var(--color-badge-blue-text)', dash: true, id: 'p' },
    { label: 'recall', ys: c.r, color: 'var(--color-badge-green-text)', dash: true, id: 'r' },
    { label: "the model's expected F1", ys: c.expected, color: 'var(--color-text)', id: 'expected' },
  ];
  return (
    <figure style={{ margin: 0 }}>
      <Caption>Thresholding span marginals: the model&rsquo;s F1 peaks at F*/2, the realized F1 need not</Caption>
      <Row>
        <Label>tagger</Label>
        {names.map((n) => (
          <Pill key={n} active={name === n} onClick={() => setName(n)} data={{ 'data-model': n }}>
            {n}{TH[n].unk ? ' ([UNK] reading)' : ''}
          </Pill>
        ))}
      </Row>
      <Row>
        <Label>test</Label>
        <Pill active={test === 'seen'} onClick={() => setTest('seen')} data={{ 'data-test': 'seen' }}>seen roster</Pill>
        <Pill active={test === 'unseen'} onClick={() => setTest('unseen')} data={{ 'data-test': 'unseen' }}>held-out roster</Pill>
      </Row>
      <LineChart xs={c.tau} series={series} xLabel="threshold τ on the span marginals" cursor={tau} yDomain={[0, 1]} id="threshold"
        marks={[{ x: half, label: 'F*/2' }]} hLines={[{ y: c.viterbi_f1, label: 'viterbi', color: 'var(--color-text-secondary)' }]} />
      <Key items={[...series.map((s) => ({ label: s.label, color: s.color, dash: s.dash })),
        { label: 'constrained Viterbi F1', color: 'var(--color-text-secondary)', dash: true }]} />
      <Row style={{ marginTop: '0.6rem' }}>
        <Label>τ</Label>
        <input type="range" min={0} max={c.tau.length - 1} step={1} value={k} onChange={(e) => setK(Number(e.target.value))}
          aria-label="threshold" data-control="tau" style={{ flex: '1 1 10rem', minWidth: 0 }} />
        <span style={{ fontFamily: mono, fontSize: '0.8rem' }} data-readout="tau">τ = {tau.toFixed(2)}</span>
      </Row>
      <Readouts>
        <Readout label="realized F1 at τ" value={f3(c.f1[k])} id="f1-tau" emphasis />
        <Readout label="expected F1 at τ" value={f3(c.expected[k])} id="exp-tau" />
        <Readout label="F* (the model's forecast)" value={f3(c.F_star)} id="fstar" />
        <Readout label="Viterbi / F1-optimal decoding" value={`${f3(c.viterbi_f1)} / ${f3(c.f1_decoded)}`} id="decoders" />
      </Readouts>
      <div style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.6rem 0 0.25rem' }}>
        held-out example: candidate spans with marginal ≥ 0.01 (kept when above τ)
      </div>
      <div data-panel="marginals" style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem 0.8rem', fontFamily: mono, fontSize: '0.75rem' }}>
        {ex.map(([s, e, ty, q]) => (
          <span key={`${s}-${e}-${ty}`} data-cand={`${s}-${e}-${ty}`}
            style={{ color: q > tau ? TYPE_COLORS[ty] : 'var(--color-text-secondary)', fontWeight: q > tau ? 600 : 400 }}>
            {ENC.tokens.slice(s, e).join(' ')} {ty} {f3(q)}
          </span>
        ))}
      </div>
      <SpanLine tokens={ENC.tokens} spans={chosen.map(([s, e, ty]) => [s, e, ty] as Span)} id="chosen" />
      <Note>
        The expected-F1 curve is the model&rsquo;s belief, 2Σq/(|Ŝ| + m); Theorem 3 puts its maximum at F*/2. Where the
        realized curve peaks elsewhere, the marginals are miscalibrated: the substitution tagger is overconfident, the
        headline underconfident under the name shift.
      </Note>
    </figure>
  );
}

const VIEWS = [
  { key: 'encodings', label: 'Encodings' },
  { key: 'heldout', label: 'Held-out names' },
  { key: 'dropout', label: 'Dropout' },
  { key: 'heads', label: 'Heads & constraints' },
  { key: 'f1', label: 'F1 decoding' },
] as const;

export default function NamedEntityLab() {
  const [view, setView] = useState<(typeof VIEWS)[number]['key']>('heldout');
  return (
    <div data-lab="named-entity" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <Row style={{ marginBottom: '0.9rem' }}>
        {VIEWS.map((v) => (
          <Pill key={v.key} active={view === v.key} onClick={() => setView(v.key)} data={{ 'data-tab': v.key }}>{v.label}</Pill>
        ))}
      </Row>
      {view === 'encodings' && <EncodingsView />}
      {view === 'heldout' && <HeldOutView />}
      {view === 'dropout' && <DropoutView />}
      {view === 'heads' && <HeadsView />}
      {view === 'f1' && <F1View />}
    </div>
  );
}
