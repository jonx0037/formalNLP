import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in
// notebooks/distributional-semantics-ppmi-svd/distributional_semantics_ppmi_svd.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and
// test_viz_payload asserts the claims the panels make. Change the .py, regenerate, never the reverse.
import payload from './data/distributional-lab.json';

/**
 * Distributional Semantics Laboratory — two linked panels for the distributional-semantics-ppmi-svd topic.
 *   A. "The company a word keeps": a word's five nearest neighbors for every context × weighting (counts, PPMI,
 *      SPPMI_{0.75,5}, the dev-selected SVD), each tagged with its planted sector or stand-in lexicon category, with the
 *      setting's test-noun AUC and nearest-neighbor accuracy, and Turney's SO accuracy for polarity words.
 *   B. "Rank and spectrum": the selected SVD's test scores against the rank d for each eigenvalue weight p; the
 *      two-word window's singular values, real against shuffled (Prop 3); its signed eigenvalues (Thm 2); and
 *      Thm 3's ratio and PPMI spectra for the document-level nouns.
 */

type Neighbor = [string, number];
type Cell = { test_auc: number; test_nn: number; neighbors: Record<string, Neighbor[]> };
type Sweep = {
  context: string; base: string; dims: number[]; sparse_auc: number; sparse_nn: number;
  by_p: Record<string, { auc: number[]; nn: number[] }>;
};
type Spectrum = Record<'ppmi' | 'sppmi5', { real: number[]; null: number[] }>;
type LowRank = { ratio: number[]; ppmi: number[]; rank: number; ratio_resid: number; ppmi_resid: number; noise: number };

const CONTEXTS = payload.contexts as string[];
const WEIGHTINGS = payload.weightings as Weighting[];
const WORDS = payload.words as string[];
const TAGS = payload.tags as Record<string, string>;
const CELLS = payload.cells as unknown as Record<string, Cell>;
const SELECTED = payload.selected as unknown as Record<'sparse' | 'svd', [string, string, number | null, number | null]>;
const CHANCE = payload.chance_nn as number;
const SO_ACC = payload.so_accuracy as Record<string, number>;
const SO_COV = payload.so_coverage as Record<string, number>;
const SWEEP = payload.sweep as unknown as Sweep;
const SPECTRUM = payload.spectrum as unknown as Spectrum;
const EIG = payload.eig_l2 as number[];
const LOWRANK = payload.lowrank as unknown as LowRank;

type Weighting = 'counts' | 'ppmi' | 'sppmi5' | 'svd';
const [, SVD_BASE, SVD_D, SVD_P] = SELECTED.svd;
const WEIGHT_LABEL: Record<Weighting, string> = {
  counts: 'raw counts', ppmi: 'PPMI', sppmi5: 'SPPMI, α = 0.75, shift log 5', svd: `SVD, d = ${SVD_D}, p = ${SVD_P}`,
};
const CONTEXT_LABEL: Record<string, string> = {
  L1: 'window 1', L2: 'window 2', L4: 'window 4', L8: 'window 8', sentence: 'sentence', document: 'document (oracle)',
};
const BASE_LABEL: Record<string, string> = { ppmi: 'PPMI', ppmi_a: 'smoothed PPMI', sppmi5: 'SPPMI (α = 0.75, k = 5)' };

// A neighbor's tag, drawn as a labeled chip: sectors filled, lexicon categories outlined.
const TAG_STYLE: Record<string, { bg: string; fg: string; outline?: boolean }> = {
  industrials: { bg: 'var(--color-badge-blue-bg)', fg: 'var(--color-badge-blue-text)' },
  technology: { bg: 'var(--color-badge-green-bg)', fg: 'var(--color-badge-green-text)' },
  banking: { bg: 'var(--color-badge-amber-bg)', fg: 'var(--color-badge-amber-text)' },
  energy: { bg: 'var(--color-badge-red-bg)', fg: 'var(--color-badge-red-text)' },
  retail: { bg: 'var(--color-definition-bg)', fg: 'var(--color-definition-border)' },
  healthcare: { bg: 'var(--color-theorem-bg)', fg: 'var(--color-text)' },
  common: { bg: 'var(--color-muted-bg)', fg: 'var(--color-text-secondary)' },
  positive: { bg: 'var(--color-surface)', fg: 'var(--color-badge-green-text)', outline: true },
  negative: { bg: 'var(--color-surface)', fg: 'var(--color-badge-red-text)', outline: true },
  uncertainty: { bg: 'var(--color-surface)', fg: 'var(--color-badge-amber-text)', outline: true },
  litigious: { bg: 'var(--color-surface)', fg: 'var(--color-badge-blue-text)', outline: true },
  polyseme: { bg: 'var(--color-surface)', fg: 'var(--color-accent)', outline: true },
};
const SECTOR_TAGS = ['industrials', 'technology', 'banking', 'energy', 'retail', 'healthcare'];
const P_STYLE: Record<string, { color: string; dash?: string; label: string }> = {
  '0.0': { color: 'var(--color-accent)', label: 'p = 0' },
  '0.5': { color: 'var(--color-accent-secondary)', dash: '6 3', label: 'p = ½' },
  '1.0': { color: 'var(--color-badge-blue-text)', dash: '2 3', label: 'p = 1' },
};

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';

const tagOf = (w: string) => TAGS[w] ?? 'other';

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

function Chip({ tag }: { tag: string }) {
  const s = TAG_STYLE[tag];
  if (!s) return <span style={{ fontFamily: sans, fontSize: '0.68rem', color: 'var(--color-text-secondary)' }}>—</span>;
  return (
    <span data-tag={tag} style={{
      fontFamily: sans, fontSize: '0.68rem', padding: '0.05rem 0.4rem', borderRadius: '0.3rem', whiteSpace: 'nowrap',
      background: s.bg, color: s.fg, border: `1px solid ${s.outline ? s.fg : 'transparent'}`,
    }}>
      {tag}
    </span>
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

function Row({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.45rem', ...style }}>{children}</div>;
}

// Panel B geometry.
const RW = 600, RH = 240, RM = { top: 20, right: 16, bottom: 34, left: 46 };
const SW = 600, SH = 210, SM = { top: 20, right: 16, bottom: 30, left: 46 };
const EW = 600, EH = 130, EM = { top: 18, right: 16, bottom: 22, left: 46 };
const LW = 600, LH = 170, LM = { top: 20, right: 16, bottom: 30, left: 46 };

export default function DistributionalLab() {
  const [word, setWord] = useState('strong');
  const [context, setContext] = useState('L2');
  const [weighting, setWeighting] = useState<Weighting>('ppmi');
  const [view, setView] = useState<'rank' | 'spectrum'>('spectrum');
  const [metric, setMetric] = useState<'auc' | 'nn'>('auc');
  const [specKey, setSpecKey] = useState<'ppmi' | 'sppmi5'>('ppmi');

  const cell = CELLS[`${context}|${weighting}`];
  const nbrs = cell.neighbors[word];
  const polar = ['positive', 'negative'].includes(tagOf(word));

  // The link: a weighting chosen in A picks the view of B that is about it.
  const pickWeighting = (w: Weighting) => {
    setWeighting(w);
    if (w === 'svd') setView('rank');
    if (w === 'ppmi' || w === 'sppmi5') {
      setView('spectrum');
      setSpecKey(w);
    }
  };

  // Panel B, rank view — test score against d (log axis), one line per eigenvalue weight p.
  const rank = useMemo(() => {
    const vals = Object.values(SWEEP.by_p).flatMap((r) => r[metric]);
    const base = metric === 'auc' ? SWEEP.sparse_auc : SWEEP.sparse_nn;
    const lo = Math.floor(Math.min(d3.min(vals)!, base) * 10) / 10;
    const x = d3.scaleLog().domain([SWEEP.dims[0], SWEEP.dims[SWEEP.dims.length - 1]]).range([0, RW - RM.left - RM.right]);
    const y = d3.scaleLinear().domain([lo, 1]).range([RH - RM.top - RM.bottom, 0]);
    return { x, y, base };
  }, [metric]);
  const rankLine = d3.line<number>().x((_, i) => rank.x(SWEEP.dims[i])).y((v) => rank.y(v));
  const selRow = SWEEP.by_p[String(Number(SVD_P).toFixed(1))];
  const selVal = selRow[metric][SWEEP.dims.indexOf(SVD_D as number)];

  // Panel B, spectrum view — singular values on a log axis, real against the shuffled corpus.
  const spec = SPECTRUM[specKey];
  const sx = d3.scaleLinear().domain([1, spec.real.length]).range([0, SW - SM.left - SM.right]);
  const sy = d3.scaleLog().domain([d3.min([...spec.real, ...spec.null])! / 1.1, d3.max([...spec.real, ...spec.null])! * 1.1])
    .range([SH - SM.top - SM.bottom, 0]);
  const specLine = d3.line<number>().x((_, i) => sx(i + 1)).y((v) => sy(v));
  const eMax = d3.max(EIG, (v) => Math.abs(v))!;
  const ex = d3.scaleBand<number>().domain(d3.range(EIG.length)).range([0, EW - EM.left - EM.right]).padding(0.15);
  const ey = d3.scaleLinear().domain([-eMax, eMax]).range([EH - EM.top - EM.bottom, 0]);
  const nNeg = EIG.filter((v) => v < 0).length;
  const lx = d3.scaleLinear().domain([1, LOWRANK.ratio.length]).range([0, LW - LM.left - LM.right]);
  const ly = d3.scaleLinear().domain([0, 1]).range([LH - LM.top - LM.bottom, 0]);

  const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;

  return (
    <div data-lab="distributional" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · The company a word keeps — nearest neighbors by context and weighting
        </figcaption>
        <Row>
          {WORDS.map((w) => (
            <Pill key={w} active={w === word} onClick={() => setWord(w)} data={{ 'data-word': w }}>{w}</Pill>
          ))}
        </Row>
        <Row>
          {CONTEXTS.map((c) => (
            <Pill key={c} active={c === context} onClick={() => setContext(c)} data={{ 'data-context': c }}>{CONTEXT_LABEL[c] ?? c}</Pill>
          ))}
        </Row>
        <Row>
          {WEIGHTINGS.map((w) => (
            <Pill key={w} active={w === weighting} onClick={() => pickWeighting(w)} data={{ 'data-weighting': w }}>{WEIGHT_LABEL[w]}</Pill>
          ))}
        </Row>

        <div data-panel="neighbors" style={{ fontFamily: sans, fontSize: '0.8rem', margin: '0.6rem 0 0.4rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
            <span style={{ fontFamily: mono, fontWeight: 700 }}>{word}</span>
            <Chip tag={tagOf(word)} />
            <span style={{ color: 'var(--color-text-secondary)' }}>
              {`· ${CONTEXT_LABEL[context]} · ${WEIGHT_LABEL[weighting]}`}
            </span>
          </div>
          {nbrs.map(([n, cos], i) => (
            <div key={n} data-neighbor={i} data-neighbor-word={n}
              style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 9rem) minmax(0, 1fr) 2.6rem auto', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
              <span title={n} style={{ fontFamily: mono, fontSize: '0.78rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{n}</span>
              <span style={{ height: '0.6rem', background: 'var(--color-muted-bg)', borderRadius: '0.2rem', overflow: 'hidden' }}>
                <span style={{ display: 'block', height: '100%', width: `${Math.max(2, cos * 100)}%`, background: 'var(--color-accent)', opacity: 0.85 }} />
              </span>
              <span style={{ fontFamily: mono, fontSize: '0.75rem', textAlign: 'right' }}>{cos.toFixed(2)}</span>
              <Chip tag={tagOf(n)} />
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.5rem' }}>
          <Readout id="test-auc" label="sector-ranking AUC, test nouns" value={cell.test_auc.toFixed(3)} emphasis />
          <Readout id="test-nn" label="nearest neighbor in sector, test nouns" value={cell.test_nn.toFixed(3)} />
          <Readout id="chance" label="chance" value={CHANCE.toFixed(3)} />
          {polar && <Readout id="so-acc" label="polarity from SO, this context" value={SO_ACC[context].toFixed(3)} />}
          {polar && <Readout id="so-cov" label="polarity words with any evidence" value={SO_COV[context].toFixed(3)} />}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.5rem', alignItems: 'center', margin: '0.6rem 0 0', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
          <span>planted sectors:</span>
          {SECTOR_TAGS.map((t) => <Chip key={t} tag={t} />)}
          <span>· stand-in lexicon:</span>
          {['positive', 'negative', 'uncertainty', 'litigious', 'polyseme'].map((t) => <Chip key={t} tag={t} />)}
          <span>· common: the shared vocabulary</span>
        </div>
        <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.5rem 0 0' }}>
          Neighbors are drawn from the training words seen at least five times. The scores are for the {`${CONTEXT_LABEL[context]}`} setting
          on the 115 test nouns; the document context reads the planted sector directly, so it was never eligible for selection.
          {polar ? ' For polarity words, SO is Turney’s semantic orientation against the other stand-in polarity words: windows give no evidence, sentences point the wrong way, documents get it right.' : ''}
        </p>
      </figure>

      <hr style={{ border: 0, borderTop: '1px solid var(--color-border)', margin: '1.25rem 0' }} />

      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Rank and spectrum — where truncation helps, and what the singular values hold
        </figcaption>
        <Row>
          <Pill active={view === 'rank'} onClick={() => setView('rank')} data={{ 'data-view': 'rank' }}>score against rank d</Pill>
          <Pill active={view === 'spectrum'} onClick={() => setView('spectrum')} data={{ 'data-view': 'spectrum' }}>spectrum, real against shuffled</Pill>
        </Row>

        {view === 'rank' ? (
          <>
            <Row>
              <Pill active={metric === 'auc'} onClick={() => setMetric('auc')} data={{ 'data-metric': 'auc' }}>sector-ranking AUC</Pill>
              <Pill active={metric === 'nn'} onClick={() => setMetric('nn')} data={{ 'data-metric': 'nn' }}>nearest neighbor</Pill>
            </Row>
            <svg viewBox={`0 0 ${RW} ${RH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="rank"
              aria-label="Test score of the SVD embedding against its rank, for three eigenvalue weights, with the sparse PPMI matrix's score as a dashed line">
              <g transform={`translate(${RM.left},${RM.top})`}>
                {rank.y.ticks(5).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={RW - RM.left - RM.right} y1={rank.y(v)} y2={rank.y(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={rank.y(v) + 3} textAnchor="end" style={axisText}>{v.toFixed(1)}</text>
                  </g>
                ))}
                {SWEEP.dims.map((d) => (
                  <text key={d} x={rank.x(d)} y={RH - RM.top - RM.bottom + 14} textAnchor="middle" style={axisText}>{d}</text>
                ))}
                <line x1={0} x2={RW - RM.left - RM.right} y1={rank.y(rank.base)} y2={rank.y(rank.base)} data-series="sparse"
                  style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1.5, strokeDasharray: '4 4' }} />
                <text x={RW - RM.left - RM.right} y={rank.y(rank.base) - 5} textAnchor="end" style={axisText}>
                  {`sparse PPMI, sentence: ${rank.base.toFixed(3)}`}
                </text>
                {Object.entries(SWEEP.by_p).map(([p, row]) => (
                  <g key={p} data-p={p}>
                    <path d={rankLine(row[metric])!} style={{ fill: 'none', stroke: P_STYLE[p].color, strokeWidth: 2, strokeDasharray: P_STYLE[p].dash }} />
                    {row[metric].map((v, i) => (
                      <circle key={i} cx={rank.x(SWEEP.dims[i])} cy={rank.y(v)} r={2.5} style={{ fill: P_STYLE[p].color }} />
                    ))}
                  </g>
                ))}
                <circle cx={rank.x(SVD_D as number)} cy={rank.y(selVal)} r={weighting === 'svd' ? 7 : 5} data-series="selected"
                  style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
              </g>
              <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
                {`${metric === 'auc' ? 'sector-ranking AUC' : 'nearest-neighbor accuracy'}, test nouns · ${CONTEXT_LABEL[SWEEP.context]}, ${BASE_LABEL[SWEEP.base] ?? SWEEP.base}`}
              </text>
              <text x={RW / 2} y={RH - 4} textAnchor="middle" style={axisText}>rank d (log scale)</text>
            </svg>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem 1rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
              {Object.entries(P_STYLE).map(([p, s]) => (
                <span key={p} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
                  <svg width="22" height="6" aria-hidden="true"><line x1="0" y1="3" x2="22" y2="3" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} /></svg>
                  {s.label}
                </span>
              ))}
              <span>circled: selected on dev nouns</span>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.4rem' }}>
              <Readout id="rank-selected" label={`selected, d = ${SVD_D}, p = ${SVD_P}`} value={selVal.toFixed(3)} emphasis />
              <Readout id="rank-sparse" label="sparse PPMI, sentence" value={rank.base.toFixed(3)} />
              <Readout id="rank-d10" label="same weight, d = 10" value={selRow[metric][0].toFixed(3)} />
              <Readout id="rank-d300" label="same weight, d = 300" value={selRow[metric][selRow[metric].length - 1].toFixed(3)} />
            </div>
            <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
              The ranking gain is a sharp peak in d. With ten dimensions the leading directions are frequency and template structure,
              not sectors; with hundreds, noise dilutes the cosine, most of all at p = 0, which weights every direction equally.
              The nearest neighbor barely moves: sparse PPMI already gets it right.
            </p>
          </>
        ) : (
          <>
            <Row>
              <Pill active={specKey === 'ppmi'} onClick={() => setSpecKey('ppmi')} data={{ 'data-spectrum': 'ppmi' }}>PPMI</Pill>
              <Pill active={specKey === 'sppmi5'} onClick={() => setSpecKey('sppmi5')} data={{ 'data-spectrum': 'sppmi5' }}>SPPMI (α = 0.75, k = 5)</Pill>
            </Row>
            <svg viewBox={`0 0 ${SW} ${SH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="spectrum"
              aria-label="Singular values of the two-word window's matrix on a log scale, for the real corpus and for the same tokens shuffled">
              <g transform={`translate(${SM.left},${SM.top})`}>
                {sy.ticks(4).filter((v) => Number.isInteger(Math.log10(v)) || [20, 50, 200].includes(v)).map((v) => (
                  <g key={v}>
                    <line x1={0} x2={SW - SM.left - SM.right} y1={sy(v)} y2={sy(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={sy(v) + 3} textAnchor="end" style={axisText}>{d3.format('~g')(v)}</text>
                  </g>
                ))}
                {[1, 25, 50, 75, 100].map((i) => (
                  <text key={i} x={sx(i)} y={SH - SM.top - SM.bottom + 14} textAnchor="middle" style={axisText}>{i}</text>
                ))}
                <path d={specLine(spec.null)!} data-series="null" style={{ fill: 'none', stroke: 'var(--color-text-secondary)', strokeWidth: 1.5, strokeDasharray: '5 3' }} />
                <path d={specLine(spec.real)!} data-series="real" style={{ fill: 'none', stroke: 'var(--color-accent)', strokeWidth: 2 }} />
              </g>
              <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
                {`singular values, window 2, ${specKey === 'ppmi' ? 'PPMI' : 'SPPMI (α = 0.75, k = 5)'} — solid: real text`}
              </text>
              <text x={SW - 8} y={12} textAnchor="end" style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-text-secondary)' }}>
                dashed: tokens shuffled
              </text>
            </svg>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.25rem' }}>
              <Readout id="sigma1-real" label="largest singular value, real" value={spec.real[0].toFixed(1)} emphasis />
              <Readout id="sigma1-null" label="largest singular value, shuffled" value={spec.null[0].toFixed(1)} />
            </div>

            <svg viewBox={`0 0 ${EW} ${EH}`} style={{ width: '100%', height: 'auto', marginTop: '0.6rem' }} role="img" data-panel="eigenvalues"
              aria-label={`The 100 eigenvalues of largest magnitude of the two-word PPMI matrix, as bars above and below zero; ${nNeg} are negative`}>
              <g transform={`translate(${EM.left},${EM.top})`}>
                <line x1={0} x2={EW - EM.left - EM.right} y1={ey(0)} y2={ey(0)} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1 }} />
                {[-eMax, eMax].map((v) => (
                  <text key={v} x={-6} y={ey(v) + 3} textAnchor="end" style={axisText}>{v.toFixed(0)}</text>
                ))}
                {EIG.map((v, i) => (
                  <rect key={i} x={ex(i)} width={ex.bandwidth()} y={Math.min(ey(v), ey(0))} height={Math.abs(ey(v) - ey(0))} data-eig={v < 0 ? 'neg' : 'pos'}
                    style={{ fill: v < 0 ? 'var(--color-accent-secondary)' : 'var(--color-accent)' }} />
                ))}
              </g>
              <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
                {`eigenvalues of window-2 PPMI by magnitude: ${nNeg} of ${EIG.length} negative (gold)`}
              </text>
            </svg>

            <svg viewBox={`0 0 ${LW} ${LH}`} style={{ width: '100%', height: 'auto', marginTop: '0.6rem' }} role="img" data-panel="lowrank"
              aria-label="Normalized singular values of the ratio matrix and of PPMI for frequent nouns at the document level, with the planted rank marked">
              <g transform={`translate(${LM.left},${LM.top})`}>
                {[0, 0.5, 1].map((v) => (
                  <g key={v}>
                    <line x1={0} x2={LW - LM.left - LM.right} y1={ly(v)} y2={ly(v)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
                    <text x={-6} y={ly(v) + 3} textAnchor="end" style={axisText}>{v}</text>
                  </g>
                ))}
                <line x1={lx(LOWRANK.rank + 0.5)} x2={lx(LOWRANK.rank + 0.5)} y1={0} y2={LH - LM.top - LM.bottom}
                  style={{ stroke: 'var(--color-text-secondary)', strokeDasharray: '3 3' }} />
                <line x1={0} x2={LW - LM.left - LM.right} y1={ly(LOWRANK.noise)} y2={ly(LOWRANK.noise)} data-series="noise"
                  style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '1 3' }} />
                {(['ppmi', 'ratio'] as const).map((k) => (
                  <g key={k} data-series={k}>
                    <path d={d3.line<number>().x((_, i) => lx(i + 1)).y((v) => ly(v))(LOWRANK[k])!}
                      style={{ fill: 'none', stroke: k === 'ratio' ? 'var(--color-accent)' : 'var(--color-accent-secondary)', strokeWidth: 2, strokeDasharray: k === 'ppmi' ? '6 3' : undefined }} />
                    {LOWRANK[k].map((v, i) => (
                      <circle key={i} cx={lx(i + 1)} cy={ly(v)} r={2.5} style={{ fill: k === 'ratio' ? 'var(--color-accent)' : 'var(--color-accent-secondary)' }} />
                    ))}
                  </g>
                ))}
                {d3.range(1, LOWRANK.ratio.length + 1).filter((i) => i === 1 || i % 5 === 0).map((i) => (
                  <text key={i} x={lx(i)} y={LH - LM.top - LM.bottom + 14} textAnchor="middle" style={axisText}>{i}</text>
                ))}
              </g>
              <text x={8} y={12} style={{ fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' }}>
                Theorem 3, frequent nouns at the document level: solid, ratio R − 11ᵀ; dashed gold, PPMI
              </text>
              <text x={LW - 8} y={LH - 4} textAnchor="end" style={axisText}>{`planted rank ${LOWRANK.rank} | dotted: shuffled-corpus noise`}</text>
            </svg>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem 1.5rem', marginTop: '0.25rem' }}>
              <Readout id="resid-ratio" label={`energy beyond rank ${LOWRANK.rank}, ratio`} value={`${(LOWRANK.ratio_resid * 100).toFixed(1)}%`} emphasis />
              <Readout id="resid-ppmi" label={`energy beyond rank ${LOWRANK.rank}, PPMI`} value={`${(LOWRANK.ppmi_resid * 100).toFixed(1)}%`} />
            </div>
            <p style={{ fontFamily: sans, fontSize: '0.75rem', color: 'var(--color-text-secondary)', margin: '0.4rem 0 0' }}>
              Unshifted PPMI's largest singular value is smaller than a shuffled corpus's: at this scale most of its energy is noise
              (Proposition 3). Shifting by log 5 puts the real spectrum on top. Nearly half the large eigenvalues are negative, which
              half-power word vectors misread (Theorem 2). For the document-level nouns, the ratio matrix shows the planted rank and
              PPMI smears it.
            </p>
          </>
        )}
      </figure>
    </div>
  );
}
