import { useState, type CSSProperties, type ReactNode } from 'react';
import * as d3 from 'd3';
// Every number below comes from viz_payload() in notebooks/beam-search-decoding/beam_search_decoding.py;
// test_viz_mirror_matches_payload asserts this file equals it exactly, and the harness's tests assert the claims the
// views make (test_lab_tree, test_search_error_against_model_error, test_greedy_is_eventually_periodic,
// test_the_mode_is_boilerplate). Change the .py, regenerate, never the reverse.
import payload from './data/beam-search-decoding-lab.json';

/**
 * Beam-Search Laboratory — three figures for the beam-search-decoding topic.
 *   A. "The search tree": the lab tree after "revenue", stepped through greedy decoding, beam search with width 2 or 3
 *      (kept, finished, and pruned prefixes at each step, and Theorem 3's certificate at the end), or uniform-cost search
 *      (its pops in order, Theorem 1).
 *   B. "Search error against model error": both corpus engines across beam widths 1–64 at α = 0, 0.5, 1 — search error
 *      (α = 0), the certified share, mean output length against the mode's and the text's, and verbatim copying.
 *   C. "Where decoding ends up": the trigram's greedy loop (Theorem 5), and six prompts' greedy, beam, normalized beam,
 *      and exact outputs on either engine.
 */

type Node = { id: string; parent: string | null; w: string; p: number; P: number };
type Step = { kept: string[]; finished: string[]; pruned: string[] };
type BeamTrace = { k: number; steps: Step[]; result: string; P: number; certified: boolean };
const TREE = payload.tree as unknown as {
  nodes: Node[]; beams: BeamTrace[]; ucs: { id: string; P: number }[]; mode: string; P_mode: number;
};
type Curve = {
  search_error: (number | null)[]; exact: (number | null)[]; certified: number[]; finished: number[]; length: number[];
  copied: number[]; distinct: number[]; nll: number[];
};
type Engine = Record<string, Curve> & { mode: { length: number; empty: number; copied: number; nll: number }; reference_length: number };
const ENGINES = { trigram: payload.trigram as unknown as Engine, decoder: payload.decoder as unknown as Engine };
const KS = payload.ks as number[];
const ALPHAS = payload.alphas as string[];
const LOOPS = payload.loops as { examples: { prompt: string; pre: string[]; cycle: string[]; period: number }[]; share: number };
type Example = { prompt: string; mode: string; greedy: string | null; beam: string | null; norm: string | null };
const EXAMPLES = payload.examples as unknown as Record<'trigram' | 'decoder', Example[]>;
const ATTRACTOR = payload.attractor as { tail: string; share: number };

const sans = 'var(--font-sans)';
const mono = 'var(--font-mono)';
const axisText = { fontFamily: sans, fontSize: '10px', fill: 'var(--color-text-secondary)' } as const;
const titleText = { fontFamily: sans, fontSize: '10.5px', fill: 'var(--color-accent)' } as const;
const f3 = (v: number) => v.toFixed(3);
const pct = (v: number) => v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');

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

function Legend({ items }: { items: { label: string; color: string; dash?: string; dot?: boolean }[] }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem 0.9rem', fontFamily: sans, fontSize: '0.72rem', color: 'var(--color-text-secondary)' }}>
      {items.map((s) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
          <svg width="22" height="8" aria-hidden="true">
            {s.dot
              ? <circle cx="11" cy="4" r="3.2" style={{ fill: s.color }} />
              : <line x1="0" y1="4" x2="22" y2="4" style={{ stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />}
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
      <label htmlFor={`bs-lab-${id}`} style={{ minWidth: 0 }}>{label}</label>
      <input id={`bs-lab-${id}`} type="range" min={0} max={max} step={1} value={value} data-control={id}
        onChange={(e) => onChange(Number(e.target.value))} style={{ flex: '1 1 10rem', minWidth: 0 }} aria-valuetext={text} />
      <span data-readout={`${id}-value`} style={{ fontFamily: mono, minWidth: 0 }}>{text}</span>
    </div>
  );
}

// ----- Figure A: the search tree -----

const TW = 600, TH = 600, TM = { top: 34, right: 30, bottom: 18, left: 34 };
const key = (id: string) => `n:${id}`;
const ROOT = d3.stratify<Node>().id((d) => key(d.id)).parentId((d) => (d.parent === null ? null : key(d.parent)))(TREE.nodes);
ROOT.sort((a, b) => b.data.p - a.data.p || d3.ascending(a.data.w, b.data.w));
d3.tree<Node>().size([TH - TM.top - TM.bottom, TW - TM.left - TM.right]).separation(() => 1)(ROOT);
const LAID = ROOT.descendants() as (d3.HierarchyPointNode<Node>)[];

type Algo = 'greedy' | 'beam2' | 'beam3' | 'ucs';
type Status = 'kept' | 'finished' | 'pruned' | 'popped' | 'current' | 'idle';

/** Node statuses after `t` steps of an algorithm (t = 0 shows the tree alone). */
function statuses(algo: Algo, t: number): Map<string, Status> {
  const out = new Map<string, Status>();
  if (t === 0) return out;
  if (algo === 'ucs') {
    TREE.ucs.slice(0, t).forEach((u, i) => out.set(u.id, i === t - 1 ? 'current' : 'popped'));
    return out;
  }
  const trace = TREE.beams[{ greedy: 0, beam2: 1, beam3: 2 }[algo]];
  trace.steps.slice(0, t).forEach((s, i) => {
    s.pruned.forEach((id) => out.set(id, 'pruned'));
    s.finished.forEach((id) => out.set(id, 'finished'));
    s.kept.forEach((id) => out.set(id, i === t - 1 ? 'current' : 'kept'));
  });
  return out;
}

const STATUS_STYLE: Record<Status, { fill: string; stroke: string; text: string; opacity: number }> = {
  idle: { fill: 'var(--color-surface)', stroke: 'var(--color-border)', text: 'var(--color-text-secondary)', opacity: 0.85 },
  kept: { fill: 'var(--color-surface)', stroke: 'var(--color-accent)', text: 'var(--color-text)', opacity: 1 },
  current: { fill: 'var(--color-accent)', stroke: 'var(--color-accent)', text: 'var(--color-text)', opacity: 1 },
  popped: { fill: 'var(--color-surface)', stroke: 'var(--color-accent)', text: 'var(--color-text)', opacity: 1 },
  finished: { fill: 'var(--color-badge-green-text)', stroke: 'var(--color-badge-green-text)', text: 'var(--color-text)', opacity: 1 },
  pruned: { fill: 'var(--color-surface)', stroke: 'var(--color-text-secondary)', text: 'var(--color-text-secondary)', opacity: 0.4 },
};

function TreeView({ algo, t }: { algo: Algo; t: number }) {
  const st = statuses(algo, t);
  return (
    <svg viewBox={`0 0 ${TW} ${TH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel="tree"
      aria-label={`The search tree after "revenue", ${algo} at step ${t}`}>
      <text x={8} y={14} style={titleText}>after “revenue”: each edge’s probability; each prefix’s probability below it</text>
      <g transform={`translate(${TM.left},${TM.top})`}>
        {LAID.filter((n) => n.parent).map((n) => {
          const s = st.get(n.data.id) ?? 'idle';
          return (
            <g key={`e${n.data.id}`} style={{ opacity: STATUS_STYLE[s].opacity }}>
              <path d={`M${n.parent!.y},${n.parent!.x} C${(n.parent!.y + n.y) / 2},${n.parent!.x} ${(n.parent!.y + n.y) / 2},${n.x} ${n.y},${n.x}`}
                style={{ fill: 'none', stroke: s === 'idle' ? 'var(--color-border)' : STATUS_STYLE[s].stroke, strokeWidth: s === 'idle' ? 1 : 1.6, strokeDasharray: s === 'pruned' ? '3 3' : undefined }} />
              <text x={(n.parent!.y + n.y) / 2} y={(n.parent!.x + n.x) / 2 - 3} textAnchor="middle" style={{ ...axisText, fontSize: '8.5px', fontFamily: mono }}>
                {n.data.p}
              </text>
            </g>
          );
        })}
        {LAID.map((n) => {
          const s = st.get(n.data.id) ?? 'idle';
          const S = STATUS_STYLE[s];
          return (
            <g key={`n${n.data.id}`} data-node={n.data.id} data-status={s} transform={`translate(${n.y},${n.x})`} style={{ opacity: S.opacity }}>
              <circle r={4.2} style={{ fill: S.fill, stroke: S.stroke, strokeWidth: 1.5 }} />
              <text x={0} y={-7} textAnchor="middle" style={{ fontFamily: sans, fontSize: '10px', fill: S.text, fontWeight: s === 'current' ? 700 : 400 }}>
                {n.data.w}
              </text>
              {n.depth > 0 && (
                <text x={0} y={13} textAnchor="middle" style={{ ...axisText, fontSize: '8.5px', fontFamily: mono }}>{n.data.P}</text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}

// ----- Figure B: curves across beam widths -----

const CW = 600, CH = 230, CM = { top: 24, right: 14, bottom: 34, left: 44 };
const CIW = CW - CM.left - CM.right, CIH = CH - CM.top - CM.bottom;
const xK = d3.scalePoint<number>().domain(KS).range([0, CIW]);

function KChart({ id, title, yMax, yLabel, series, hlines = [], sel, aria }: {
  id: string; title: string; yMax: number; yLabel: string;
  series: { key: string; ys: (number | null)[]; color: string; dash?: string }[];
  hlines?: { v: number; label: string }[]; sel: number; aria: string;
}) {
  const y = d3.scaleLinear().domain([0, yMax]).range([CIH, 0]);
  const line = d3.line<[number, number]>().x((p) => xK(p[0])!).y((p) => y(p[1]));
  return (
    <svg viewBox={`0 0 ${CW} ${CH}`} style={{ width: '100%', height: 'auto' }} role="img" data-panel={id} aria-label={aria}>
      <text x={8} y={14} style={titleText}>{title}</text>
      <g transform={`translate(${CM.left},${CM.top})`}>
        {y.ticks(4).map((t) => (
          <g key={`y${t}`}>
            <line x1={0} x2={CIW} y1={y(t)} y2={y(t)} style={{ stroke: 'var(--color-border)', strokeWidth: 0.5 }} />
            <text x={-5} y={y(t) + 3} textAnchor="end" style={axisText}>{String(Number(t.toFixed(2)))}</text>
          </g>
        ))}
        {KS.map((k) => (
          <text key={`x${k}`} x={xK(k)} y={CIH + 14} textAnchor="middle" style={{ ...axisText, fontWeight: k === KS[sel] ? 700 : 400 }}>{k}</text>
        ))}
        <line x1={xK(KS[sel])} x2={xK(KS[sel])} y1={0} y2={CIH} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '2 3' }} />
        {hlines.map((h) => (
          <g key={`h${h.label}`}>
            <line x1={0} x2={CIW} y1={y(h.v)} y2={y(h.v)} style={{ stroke: 'var(--color-text-secondary)', strokeWidth: 1, strokeDasharray: '5 3' }} />
            <text x={CIW - 4} y={y(h.v) - 3} textAnchor="end" style={axisText}>{h.label}</text>
          </g>
        ))}
        {series.map((s) => {
          const pts = KS.map((k, i) => [k, s.ys[i]] as [number, number | null]).filter((p): p is [number, number] => p[1] !== null);
          return (
            <g key={s.key} data-series={s.key}>
              <path d={line(pts) ?? ''} style={{ fill: 'none', stroke: s.color, strokeWidth: 2, strokeDasharray: s.dash }} />
              {pts.map((p) => <circle key={p[0]} cx={xK(p[0])} cy={y(p[1])} r={2.8} style={{ fill: s.color }} />)}
            </g>
          );
        })}
      </g>
      <text x={CM.left + CIW / 2} y={CH - 4} textAnchor="middle" style={axisText}>beam width k (k = 1 is greedy decoding)</text>
      <text transform={`translate(11,${CM.top + CIH / 2}) rotate(-90)`} textAnchor="middle" style={axisText}>{yLabel}</text>
    </svg>
  );
}

// ----- Figure C pieces -----

function Words({ text, emphasis }: { text: string | null; emphasis?: boolean }) {
  if (text === null) return <span style={{ fontFamily: mono, color: 'var(--color-text-secondary)' }}>never ends (cycles)</span>;
  return <span style={{ fontFamily: mono, color: emphasis ? 'var(--color-accent)' : 'var(--color-text)', overflowWrap: 'anywhere' }}>{text}</span>;
}

const ALGO_LABELS: Record<Algo, string> = { greedy: 'greedy', beam2: 'beam, k = 2', beam3: 'beam, k = 3', ucs: 'uniform-cost' };

export default function BeamSearchLab() {
  const [algo, setAlgo] = useState<Algo>('beam2');
  const [t, setT] = useState(0);
  const [engine, setEngine] = useState<'trigram' | 'decoder'>('trigram');
  const [alpha, setAlpha] = useState('0');
  const [kIdx, setKIdx] = useState(4);
  const [exEngine, setExEngine] = useState<'trigram' | 'decoder'>('trigram');

  // ----- Figure A -----
  const trace = algo === 'ucs' ? null : TREE.beams[{ greedy: 0, beam2: 1, beam3: 2 }[algo]];
  const T = trace ? trace.steps.length : TREE.ucs.length;
  const tt = Math.min(t, T);
  const done = tt === T;
  const prunedIds = trace ? trace.steps.flatMap((s) => s.pruned) : [];
  const P = new Map(TREE.nodes.map((n) => [n.id, n.P]));
  const bestPruned = prunedIds.length ? d3.max(prunedIds, (id) => P.get(id)!)! : null;
  const stepNow = trace && tt > 0 ? trace.steps[tt - 1] : null;
  const popNow = !trace && tt > 0 ? TREE.ucs[tt - 1] : null;
  const pick = (a: Algo) => { setAlgo(a); setT(0); };

  // ----- Figure B -----
  const E = ENGINES[engine];
  const C = E[alpha];
  const exact = alpha === '0';

  return (
    <div data-lab="beam-search" style={{ border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '1rem', margin: '2rem 0' }}>
      <figure style={{ margin: 0 }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          A · The search tree — greedy, beam, and uniform-cost search, one step at a time
        </figcaption>
        <Row>
          {(['greedy', 'beam2', 'beam3', 'ucs'] as Algo[]).map((a) => (
            <Pill key={a} active={algo === a} onClick={() => pick(a)} data={{ 'data-algo': a }}>{ALGO_LABELS[a]}</Pill>
          ))}
        </Row>
        <Slider id="step" label="step" value={tt} max={T} onChange={setT} text={`${tt} of ${T}`} />
        <TreeView algo={algo} t={tt} />
        <Legend items={trace ? [
          { label: 'kept this step', color: 'var(--color-accent)', dot: true },
          { label: 'kept earlier', color: 'var(--color-accent)' },
          { label: 'finished', color: 'var(--color-badge-green-text)', dot: true },
          { label: 'pruned', color: 'var(--color-text-secondary)', dash: '3 3' },
        ] : [
          { label: 'popped this step', color: 'var(--color-accent)', dot: true },
          { label: 'popped earlier', color: 'var(--color-accent)' },
        ]} />
        <Readouts>
          {trace && stepNow && (
            <>
              <Readout id="kept" label="kept" value={stepNow.kept.length ? stepNow.kept.join(' · ') : '—'} />
              <Readout id="finished" label="finished this step" value={stepNow.finished.length ? stepNow.finished.join(' · ') : '—'} />
              <Readout id="pruned" label="pruned this step" value={stepNow.pruned.length ? stepNow.pruned.join(' · ') : '—'} />
            </>
          )}
          {popNow && <Readout id="pop" label="popped" value={`${popNow.id || 'revenue (root)'} — ${pct(popNow.P)}`} />}
          {done && trace && (
            <>
              <Readout id="result" label="returned" value={`${trace.result} — ${pct(trace.P)}`} emphasis />
              <Readout id="best-pruned" label="best pruned prefix" value={bestPruned === null ? '—' : pct(bestPruned)} />
              <Readout id="certified" label="certificate (Theorem 3)" value={trace.certified ? 'fires: the mode' : 'silent'} />
            </>
          )}
          {done && !trace && <Readout id="result" label="returned (Theorem 1: the mode)" value={`${TREE.mode} — ${pct(TREE.P_mode)}`} emphasis />}
        </Readouts>
        <Note>
          Greedy decoding commits to “rose” and never sees “was flat”, the mode at {pct(TREE.P_mode)}. Width 2 keeps “was”,
          finds the mode, and proves it: the best prefix it pruned, “fell”, had less probability than its answer. Uniform-cost
          search pops prefixes in order of probability and stops at the first finished string.
        </Note>
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          B · Search error against model error — {payload.n_prompts} MD&amp;A prompts, beam widths 1 to 64
        </figcaption>
        <Row>
          <Pill active={engine === 'trigram'} onClick={() => setEngine('trigram')} data={{ 'data-engine': 'trigram' }}>Kneser–Ney trigram</Pill>
          <Pill active={engine === 'decoder'} onClick={() => setEngine('decoder')} data={{ 'data-engine': 'decoder' }}>toy transformer</Pill>
        </Row>
        <Row>
          {ALPHAS.map((a) => (
            <Pill key={a} active={alpha === a} onClick={() => setAlpha(a)} data={{ 'data-alpha': a }}>
              {a === '0' ? 'probability (α = 0)' : `length-normalized, α = ${a}`}
            </Pill>
          ))}
        </Row>
        <Slider id="k" label="beam width k" value={kIdx} max={KS.length - 1} onChange={setKIdx} text={String(KS[kIdx])} />
        <KChart id="errors" title={exact ? 'share of prompts: beam misses the mode, finds it, certifies it' : 'share of prompts on which the length-capped certificate fires'}
          yMax={1} yLabel="share of prompts" sel={kIdx}
          series={exact ? [
            { key: 'search_error', ys: C.search_error, color: 'var(--color-accent)' },
            { key: 'exact', ys: C.exact, color: 'var(--color-badge-blue-text)' },
            { key: 'certified', ys: C.certified, color: 'var(--color-badge-green-text)', dash: '5 3' },
          ] : [{ key: 'certified', ys: C.certified, color: 'var(--color-badge-green-text)', dash: '5 3' }]}
          aria={`${engine}: at width ${KS[kIdx]}, certified share ${f3(C.certified[kIdx])}`} />
        <Legend items={exact ? [
          { label: 'search error (misses the mode)', color: 'var(--color-accent)' },
          { label: 'returns the mode', color: 'var(--color-badge-blue-text)' },
          { label: 'certified (Theorem 3)', color: 'var(--color-badge-green-text)', dash: '5 3' },
        ] : [{ label: 'certified (Proposition 2)', color: 'var(--color-badge-green-text)', dash: '5 3' }]} />
        <KChart id="length" title="mean words before </s>, finished outputs only" yMax={8} yLabel="words" sel={kIdx}
          series={[{ key: 'length', ys: C.length, color: 'var(--color-accent)' }]}
          hlines={[{ v: E.reference_length, label: `held-out text ${f3(E.reference_length)}` }, { v: E.mode.length, label: `exact mode (α = 0) ${f3(E.mode.length)}` }]}
          aria={`${engine}: mean output length ${f3(C.length[kIdx])} at width ${KS[kIdx]}, mode ${f3(E.mode.length)}, text ${f3(E.reference_length)}`} />
        <Readouts>
          {exact && <Readout id="search-error" label="search error" value={f3(C.search_error[kIdx]!)} emphasis />}
          <Readout id="certified-share" label="certified" value={f3(C.certified[kIdx])} emphasis={!exact} />
          <Readout id="length" label="mean length (words)" value={f3(C.length[kIdx])} />
          <Readout id="finished" label="outputs that end" value={f3(C.finished[kIdx])} />
          <Readout id="copied" label="a training sentence verbatim" value={f3(C.copied[kIdx])} />
          <Readout id="nll" label="nats per word" value={f3(C.nll[kIdx])} />
        </Readouts>
        <Note>
          {engine === 'trigram'
            ? 'At α = 0: the trigram’s prefixes recombine, so a beam of 64 always reaches the mode, and always proves it; its outputs shrink to the mode’s length, a third of the text’s. Greedy decoding cycles on a share of the prompts, so its length counts only the outputs that end.'
            : 'At α = 0: nothing recombines in the transformer, so even a beam of 64 misses the mode on a fifth of the prompts. Its outputs shorten as the beam widens, toward a mode that is never empty but still shorter than the text.'}
          {!exact && ' Under normalization there is no exact optimum to compare against, only the certificate, which a length cap makes possible.'}
        </Note>
      </figure>

      <figure style={{ margin: '1.75rem 0 0' }}>
        <figcaption style={{ fontFamily: sans, fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          C · Where decoding ends up — a greedy loop, and six prompts decoded four ways
        </figcaption>
        <div data-panel="loops" style={{ fontFamily: sans, fontSize: '0.8rem' }}>
          {LOOPS.examples.map((e) => (
            <div key={e.prompt} data-loop={e.prompt} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 0.4rem', alignItems: 'baseline', marginBottom: '0.35rem' }}>
              <span style={{ fontWeight: 600 }}>{e.prompt}</span>
              <span style={{ fontFamily: mono, color: 'var(--color-text-secondary)', overflowWrap: 'anywhere' }}>{e.pre.join(' ')}</span>
              <span style={{ fontFamily: mono, color: 'var(--color-accent)', border: '1px solid var(--color-accent)', borderRadius: '0.4rem', padding: '0 0.3rem' }}>
                {e.cycle.join(' ')} ↻
              </span>
            </div>
          ))}
        </div>
        <Readouts>
          <Readout id="loop-share" label="trigram prompts on which greedy decoding cycles (Theorem 5)" value={f3(LOOPS.share)} emphasis />
        </Readouts>
        <Note>
          Greedy decoding on a finite-state model is a walk on its states. After “include” the likeliest word is “and”, and
          after “and and” it is “and” again: every looping prompt drains into the same one-word cycle.
        </Note>

        <div style={{ marginTop: '1.25rem' }}>
          <Row>
            <Pill active={exEngine === 'trigram'} onClick={() => setExEngine('trigram')} data={{ 'data-ex-engine': 'trigram' }}>Kneser–Ney trigram</Pill>
            <Pill active={exEngine === 'decoder'} onClick={() => setExEngine('decoder')} data={{ 'data-ex-engine': 'decoder' }}>toy transformer</Pill>
          </Row>
          <div data-panel="examples" style={{ fontFamily: sans, fontSize: '0.78rem' }}>
            {EXAMPLES[exEngine].map((e) => (
              <div key={e.prompt} style={{ borderTop: '1px solid var(--color-border)', padding: '0.4rem 0' }}>
                <div style={{ fontWeight: 600, marginBottom: '0.2rem' }}>{e.prompt} …</div>
                {([['greedy', e.greedy], ['beam, k = 16', e.beam], ['normalized beam, k = 16, α = 1', e.norm], ['exact mode', e.mode]] as [string, string | null][]).map(([label, text]) => (
                  <div key={label} style={{ display: 'flex', flexWrap: 'wrap', gap: '0 0.5rem' }}>
                    <span style={{ color: 'var(--color-text-secondary)', minWidth: 0 }}>{label}:</span>
                    <Words text={text} emphasis={label === 'exact mode'} />
                  </div>
                ))}
              </div>
            ))}
          </div>
          <Note>
            {exEngine === 'trigram'
              ? `The trigram’s mode often just stops. Normalization lengthens its outputs by steering them onto the 8-K template; at k = 16, ${f3(ATTRACTOR.share)} of all prompts end in “${ATTRACTOR.tail}”.`
              : 'The transformer’s outputs are longer and more varied, and its mode is never empty, but greedy decoding and the beam still disagree with it often.'}
          </Note>
        </div>
      </figure>
    </div>
  );
}
