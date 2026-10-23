import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/subword-tokenization-bpe-unigram/subword_tokenization_bpe_unigram.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/subword-tokenization-lab.json';

/**
 * Subword Tokenization Laboratory — two linked panels for the subword-tokenization topic.
 *   A. "Run the merges": BPE or WordPiece after 0–774 merges (|V| = 26–800). Six words drawn as
 *      piece strips, with the planted morpheme boundaries of the pseudo-words marked underneath and
 *      the Unigram LM's segmentation at the same |V| for comparison; tokens per held-out word falls
 *      from 6.348 to 1.539 (BPE) or 1.692 (WordPiece).
 *   B. "Read the lattice": one word's Unigram lattice — every piece as an arc, width ∝ its
 *      forward–backward posterior, the Viterbi path highlighted — with Proposition 7's sandwich:
 *      marginal bits ≤ Viterbi bits ≤ marginal bits + log₂|Seg|.
 * One |V| control drives both panels; the strip reports boundary F1 against the planted morphemes.
 */

type Kind = 'bpe' | 'wordpiece';
type MergeRow = { merges: number; vocab: number } & Record<Kind, { tpw: number; segs: Record<string, string[]> }>;
type Lattice = { edges: [number, number, number][]; viterbi: string[]; bits_viterbi: number; bits_marginal: number; log2_segs: number };

const MERGES = payload.merges as unknown as Record<string, MergeRow>;
const STEPS = Object.values(MERGES).sort((a, b) => a.merges - b.merges); // |V| = 26, 50, …, 800
const LATTICE = payload.lattice as unknown as Record<string, Record<string, Lattice>>;
const UNIGRAM = payload.unigram_segs as unknown as Record<string, Record<string, string[]>>;
const PLANTED = payload.planted as Record<string, number[] | null>;
const F1 = payload.f1 as unknown as Record<string, Record<'bpe' | 'wordpiece' | 'unigram', number>>;
const WORDS = payload.words;
const LATTICE_WORDS = payload.lattice_words;

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';

function boundaries(pieces: string[]): number[] {
  let at = 0;
  return pieces.slice(0, -1).map((p) => (at += p.length));
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

// Panel A geometry: a label column, then one 17-px cell per character.
const AW = 560, CELL = 17, LABEL = 118, ROW = 22, GAP = 16;
const AH = WORDS.length * (2 * ROW + GAP) + 8;

function drawStrip(
  g: d3.Selection<SVGGElement, unknown, null, undefined>, pieces: string[], y: number,
  planted: Set<number>, tone: 'primary' | 'secondary',
) {
  let at = 0;
  for (const p of pieces) {
    g.append('rect').attr('x', LABEL + at * CELL + 1).attr('y', y).attr('width', p.length * CELL - 2).attr('height', ROW - 4)
      .attr('rx', 4)
      .style('fill', tone === 'primary' ? 'var(--color-surface)' : 'var(--color-muted-bg)')
      .style('stroke', 'var(--color-muted-border)');
    g.append('text').attr('x', LABEL + (at + p.length / 2) * CELL).attr('y', y + ROW / 2 + 2).attr('text-anchor', 'middle')
      .style('font-family', mono).style('font-size', '12px')
      .style('fill', tone === 'primary' ? 'var(--color-text)' : 'var(--color-text-secondary)').text(p);
    at += p.length;
  }
  // A predicted boundary that matches a planted one is drawn in the accent color.
  for (const b of boundaries(pieces)) {
    if (!planted.has(b)) continue;
    g.append('line').attr('x1', LABEL + b * CELL).attr('x2', LABEL + b * CELL).attr('y1', y - 2).attr('y2', y + ROW - 2)
      .style('stroke', 'var(--color-accent)').style('stroke-width', 2.5).attr('data-hit', 'true');
  }
}

// Panel B geometry: characters on a baseline, lattice arcs above it.
const BW = 560, BH = 250, BX = 40, BASE = 205;

export default function SubwordTokenizationLab() {
  const [step, setStep] = useState(STEPS.findIndex((r) => r.vocab === 400));
  const [kind, setKind] = useState<Kind>('bpe');
  const [word, setWord] = useState(LATTICE_WORDS[1]);
  const stripsRef = useRef<SVGSVGElement>(null);
  const latticeRef = useRef<SVGSVGElement>(null);

  const row = STEPS[step];
  const size = String(Math.max(row.vocab, 50)); // Unigram has no 26-piece model; Panel B starts at 50
  const lat = LATTICE[size][word];
  const unigramOn = row.vocab >= 50;

  // Panel A — piece strips for the chosen merge tokenizer, with Unigram beneath.
  useEffect(() => {
    if (!stripsRef.current) return;
    const svg = d3.select(stripsRef.current);
    svg.selectAll('*').remove();
    WORDS.forEach((w, k) => {
      const y0 = 4 + k * (2 * ROW + GAP);
      const g = svg.append('g').attr('data-word', w);
      const planted = new Set(PLANTED[w] ?? []);
      g.append('text').attr('x', 0).attr('y', y0 + ROW / 2 + 2).style('font-family', mono).style('font-size', '12px')
        .style('fill', 'var(--color-text)').text(w);
      g.append('text').attr('x', 0).attr('y', y0 + ROW + ROW / 2 + 2).style('font-family', sans).style('font-size', '10px')
        .style('fill', 'var(--color-text-secondary)').text(PLANTED[w] ? 'planted morphemes' : 'real word');
      drawStrip(g, row[kind].segs[w], y0, planted, 'primary');
      if (unigramOn) drawStrip(g, UNIGRAM[size][w], y0 + ROW, planted, 'secondary');
      for (const b of planted) {
        const x = LABEL + b * CELL, y = y0 + 2 * ROW - 2;
        g.append('path').attr('d', `M${x},${y} l-4,7 l8,0 z`).style('fill', 'var(--color-accent-secondary)');
      }
    });
    svg.append('text').attr('x', AW).attr('y', AH - 2).attr('text-anchor', 'end').style('font-family', sans)
      .style('font-size', '10px').style('fill', 'var(--color-text-secondary)')
      .text(unigramOn ? `upper strip: ${kind === 'bpe' ? 'BPE' : 'WordPiece'} · lower strip: Unigram LM at the same |V|` : 'Unigram LM starts at |V| = 50');
  }, [step, kind]);

  // Panel B — the Unigram lattice of one word.
  useEffect(() => {
    if (!latticeRef.current) return;
    const svg = d3.select(latticeRef.current);
    svg.selectAll('*').remove();
    const n = word.length;
    const cw = Math.min(40, (BW - 2 * BX) / n);
    const x = (i: number) => BX + i * cw;
    const vit = new Set<string>();
    let at = 0;
    for (const p of lat.viterbi) { vit.add(`${at}-${at + p.length}`); at += p.length; }
    const maxSpan = Math.max(...lat.edges.map(([i, j]) => j - i));
    const hScale = (BASE - 16) / Math.max(1, maxSpan);
    const sorted = [...lat.edges].sort((a, b) => Number(vit.has(`${a[0]}-${a[1]}`)) - Number(vit.has(`${b[0]}-${b[1]}`)));
    svg.append('g').selectAll('path').data(sorted).join('path')
      .attr('d', ([i, j]) => `M${x(i)},${BASE} C${x(i)},${BASE - (j - i) * hScale} ${x(j)},${BASE - (j - i) * hScale} ${x(j)},${BASE}`)
      .attr('data-viterbi', ([i, j]) => String(vit.has(`${i}-${j}`)))
      .style('fill', 'none')
      .style('stroke', ([i, j]) => (vit.has(`${i}-${j}`) ? 'var(--color-accent)' : 'var(--color-text-secondary)'))
      .style('stroke-width', ([, , q]) => 1 + 6 * q)
      .style('opacity', ([i, j, q]) => (vit.has(`${i}-${j}`) ? 1 : 0.25 + 0.6 * q))
      .append('title').text(([i, j, q]) => `${word.slice(i, j)}: posterior ${q.toFixed(3)}`);
    svg.append('line').attr('x1', x(0)).attr('x2', x(n)).attr('y1', BASE).attr('y2', BASE)
      .style('stroke', 'var(--color-muted-border)');
    for (let i = 0; i <= n; i++) {
      svg.append('circle').attr('cx', x(i)).attr('cy', BASE).attr('r', 3).style('fill', 'var(--color-text-secondary)');
    }
    for (let i = 0; i < n; i++) {
      svg.append('text').attr('x', x(i) + cw / 2).attr('y', BASE + 20).attr('text-anchor', 'middle')
        .style('font-family', mono).style('font-size', '14px').style('fill', 'var(--color-text)').text(word[i]);
    }
    for (const b of PLANTED[word] ?? []) {
      svg.append('path').attr('d', `M${x(b)},${BASE + 26} l-4,8 l8,0 z`).style('fill', 'var(--color-accent-secondary)');
    }
  }, [word, size]);

  const f1 = unigramOn ? F1[size] : null;

  return (
    <div data-lab="subword-tokenization" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontFamily: sans, fontSize: '0.875rem', marginBottom: '0.75rem' }}>
        <span style={{ minWidth: '9rem' }}>vocabulary |V| = <strong data-readout="vocab">{row.vocab}</strong></span>
        <input type="range" min={0} max={STEPS.length - 1} step={1} value={step} aria-label="Vocabulary size (both panels)"
          onChange={(e) => setStep(parseInt(e.target.value, 10))} style={{ flex: 1, accentColor: 'var(--color-accent)' }} />
      </label>

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · Run the merges — {row.merges} merges on the 26 letters
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          <Pill active={kind === 'bpe'} onClick={() => setKind('bpe')} data={{ 'data-kind': 'bpe' }}>BPE (most frequent pair)</Pill>
          <Pill active={kind === 'wordpiece'} onClick={() => setKind('wordpiece')} data={{ 'data-kind': 'wordpiece' }}>WordPiece (largest likelihood gain)</Pill>
        </div>
        <svg ref={stripsRef} viewBox={`0 0 ${AW} ${AH}`} style={{ width: '100%', height: 'auto' }} role="img"
          aria-label="Six words split into subword pieces by the chosen merge tokenizer and by the Unigram LM, with planted morpheme boundaries marked" />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.25rem' }}>
          <Readout id="merges" label="merges" value={String(row.merges)} />
          <Readout id="tpw" label={`tokens per held-out word (${kind === 'bpe' ? 'BPE' : 'WordPiece'})`} value={row[kind].tpw.toFixed(3)} emphasis />
          <Readout id="tpw-other" label={`(${kind === 'bpe' ? 'WordPiece' : 'BPE'} at the same |V|)`} value={row[kind === 'bpe' ? 'wordpiece' : 'bpe'].tpw.toFixed(3)} />
        </div>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Read the lattice — the Unigram LM at |V| = {size}
        </figcaption>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.5rem' }}>
          {LATTICE_WORDS.map((w) => (
            <Pill key={w} active={w === word} onClick={() => setWord(w)} data={{ 'data-lattice-word': w }}>{w}</Pill>
          ))}
        </div>
        <svg ref={latticeRef} viewBox={`0 0 ${BW} ${BH}`} style={{ width: '100%', height: 'auto' }} role="img"
          aria-label="Segmentation lattice of the chosen word: each arc is a vocabulary piece, its width the posterior probability, the Viterbi path highlighted" />
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', marginTop: '0.25rem' }}>
          <Readout id="viterbi" label="Viterbi segmentation" value={lat.viterbi.join(' · ')} emphasis />
          <Readout id="bits-marginal" label="−log₂ P(x), marginal" value={`${lat.bits_marginal.toFixed(2)} bits`} />
          <Readout id="bits-viterbi" label="−log₂ P(t*), Viterbi" value={`${lat.bits_viterbi.toFixed(2)} bits`} />
          <Readout id="log2-segs" label="log₂ |Seg(x)|" value={lat.log2_segs.toFixed(2)} />
        </div>
      </figure>

      <div data-readout="strip" style={{
        marginTop: '1rem', padding: '0.6rem 0.8rem', borderRadius: '0.5rem', background: 'var(--color-definition-bg)',
        fontFamily: sans, fontSize: '0.85rem',
      }}>
        {f1 ? (
          <>
            Boundary F1 against the planted morphemes at |V| = {size}: BPE <strong>{f1.bpe.toFixed(3)}</strong>,
            {' '}WordPiece <strong>{f1.wordpiece.toFixed(3)}</strong>, Unigram LM <strong>{f1.unigram.toFixed(3)}</strong>.
          </>
        ) : (
          <>With only the 26 letters every boundary is a split, so recall is perfect and precision is not.</>
        )}
      </div>
      <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '0.6rem', marginBottom: 0 }}>
        Arcs below a posterior of 0.005 are not drawn. The sandwich of Proposition 7 holds for every word and size:
        the Viterbi code wastes at most log₂|Seg(x)| bits over the marginal code, and in practice a fraction of a bit.
      </p>
    </div>
  );
}
