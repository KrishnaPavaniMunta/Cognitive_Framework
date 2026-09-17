import React from 'react'
import { createRoot } from 'react-dom/client'
import { ArrowUpRight, BrainCircuit, Check, CircleDot, Cpu, Database, ExternalLink, Filter, GitBranch, Map, Menu, ScanLine, ShieldCheck, X } from 'lucide-react'
import { SiGithub } from '@icons-pack/react-simple-icons'
import './styles.css'

const PUBLIC_BASE = import.meta.env.BASE_URL

const demonstrations = [
  {
    id: 'dimensions',
    eyebrow: 'Demonstration 01',
    title: 'Objects, measured before they matter',
    description: 'Object detections are checked against estimated physical dimensions before they enter the scene model. Common-sense filters reduce implausible or unsafe interpretations.',
    video: `${PUBLIC_BASE}media/demo-object-detection.mp4`,
    tags: ['YOLO detection', 'Dimension gating', 'Common-sense filtering'],
  },
  {
    id: 'obstruction',
    eyebrow: 'Demonstration 02',
    title: 'A blocked exit becomes a spatial event',
    description: 'Door and obstruction reasoning connects visual detections to a safety-relevant event: an exit that is present, understood, and no longer clear.',
    video: `${PUBLIC_BASE}media/demo-door-obstruction.mp4`,
    tags: ['Door reasoning', 'Obstacle detection', 'Safety event'],
  },
]

const systemSteps = [
  ['01', 'Perceive', 'RGB-D sensing and trained detectors observe objects, people, doors, and context.'],
  ['02', 'Ground', 'Detector labels resolve to an ontology with aliases, dimensions, and semantic meaning.'],
  ['03', 'Remember', 'A persistent semantic map gives observations a place, history, and spatial relationship.'],
  ['04', 'Reason', 'Rules and common sense turn raw detections into interpretable safety events.'],
]

function InstitutionalMark({ type, children }) {
  const logos = { 'ou-mark': [`${PUBLIC_BASE}logos/open-university.svg`, 'The Open University'], 'kmi-mark': [`${PUBLIC_BASE}logos/kmi.svg`, 'Knowledge Media Institute'], 'resilient-mark': [`${PUBLIC_BASE}logos/resilient-enterprise.svg`, 'Resilient Enterprise'] }
  const [src, alt] = logos[type]
  return <div className={`institutional-mark ${type}`} aria-label={alt}><img src={src} alt={alt} /></div>
}

function PageHeader() {
  return <header className="site-header"><a className="brand" href="#/" aria-label="Cognitive Recognition Framework home"><span className="brand-mark">CRF</span><span><strong>Cognitive Recognition</strong><small>Framework</small></span></a><nav className="nav-links page-nav"><a href="#/research">Research</a><a href="#/architecture">Architecture</a><a href="#/demonstrations">Demonstrations</a><a href="#/ontology">Ontology</a><a href="#/robot">Robot</a><a href="#/about">About</a></nav><a className="header-link" href="https://github.com/KrishnaPavaniMunta" target="_blank" rel="noreferrer" aria-label="Krishna Pavani Munta on GitHub"><SiGithub size={18} /> GitHub</a></header>
}

function PageFrame({ eyebrow, title, intro, children }) {
  return <div className="page-shell"><PageHeader /><main><section className="page-hero section-wrap"><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p className="page-intro">{intro}</p></section>{children}</main><footer className="site-footer"><div className="footer-top"><div className="footer-brand"><span className="brand-mark">CRF</span><div><strong>Cognitive Recognition Framework</strong><small>Hospital robotics research</small></div></div><div className="institutional-marks"><InstitutionalMark type="ou-mark" /><InstitutionalMark type="kmi-mark" /><InstitutionalMark type="resilient-mark" /></div></div><div className="footer-bottom"><span>© 2026 Krishna Pavani Munta. All rights reserved.</span><span>Research website · Public project record</span><a href="https://github.com/KrishnaPavaniMunta" target="_blank" rel="noreferrer" aria-label="Krishna Pavani Munta on GitHub"><SiGithub size={17} /></a></div><p className="viewing-note">Best viewed on a laptop or computer screen.</p></footer></div>
}

function ResearchPage() {
  return <PageFrame eyebrow="Research record / 01" title={<>Recognition needs <em>context.</em></>} intro="This project studies how hospital robots can combine learned perception with explicit knowledge, physical expectations, and spatial memory to produce decisions people can inspect."><section className="section-wrap page-content"><div className="detail-grid"><article><p className="eyebrow">Research question</p><h2>What does an object mean <em>here</em>?</h2><p>Detection is only the beginning. A hospital scene contains objects, roles, locations, hazards, and relationships. The framework connects these layers so a robot can distinguish a plausible observation from an operationally meaningful event.</p></article><article className="quote-panel"><span className="quote-mark">“</span><p>A situated recognition system should be able to explain not only what it sees, but why that observation matters.</p><small>Framework principle / 2026</small></article></div><div className="steps-grid page-steps">{systemSteps.map(([number, title, copy]) => <article className="step" key={number}><span className="step-number">{number}</span><h3>{title}</h3><p>{copy}</p></article>)}</div><div className="research-columns"><div><p className="eyebrow">Core contributions</p><h3>One pipeline, several kinds of evidence.</h3></div><ul className="evidence-list"><li><strong>Visual evidence</strong><span>Detections, confidence, depth, and image context.</span></li><li><strong>Physical evidence</strong><span>Estimated dimensions and plausibility gates.</span></li><li><strong>Semantic evidence</strong><span>Ontology classes, aliases, and relations.</span></li><li><strong>Spatial evidence</strong><span>Persistent landmarks and map-relative locations.</span></li></ul></div></section></PageFrame>
}

function DemonstrationsPage() {
  return <PageFrame eyebrow="Working examples / 02" title={<>From detection to <em>decision.</em></>} intro="Two working examples show how the framework narrows raw perception into interpretable hospital-safety reasoning."><section className="dark-band page-band"><div className="section-wrap"><div className="demo-grid">{demonstrations.map((demo) => <article className="demo-card" key={demo.id}><div className="video-frame"><video controls preload="metadata" src={demo.video}><track kind="captions" /></video><span className="video-index">{demo.eyebrow}</span></div><div className="demo-copy"><p className="eyebrow">{demo.eyebrow}</p><h3>{demo.title}</h3><p>{demo.description}</p><div className="tag-list">{demo.tags.map(tag => <span key={tag}>{tag}</span>)}</div></div></article>)}</div><div className="method-strip"><span>Detection</span><b>→</b><span>Dimension gate</span><b>→</b><span>Common sense</span><b>→</b><span>Safety event</span></div></div></section></PageFrame>
}

const architectureStages = [
  { id: '01', title: 'Perceive', label: 'RGB-D + YOLO', detail: 'Images, depth, detections, confidence', icon: ScanLine, tone: 'blue' },
  { id: '02', title: 'Gate', label: 'Physical dimensions', detail: 'Plausibility checks and size estimates', icon: Filter, tone: 'rust' },
  { id: '03', title: 'Ground', label: 'Ontology / RDF', detail: 'Classes, aliases, properties, meaning', icon: GitBranch, tone: 'sage' },
  { id: '04', title: 'Remember', label: 'Semantic map', detail: 'Landmarks, locations, time, relationships', icon: Map, tone: 'blue' },
  { id: '05', title: 'Reason', label: 'Common sense', detail: 'Rules combine context and evidence', icon: BrainCircuit, tone: 'rust' },
  { id: '06', title: 'Act / report', label: 'Safety event', detail: 'Explainable alert or monitored state', icon: ShieldCheck, tone: 'sage' },
]

const ARCH_W = 1000, ARCH_H = 680
const ARCH_ICONS = { Cpu, ScanLine, Filter, GitBranch, Map, BrainCircuit, ShieldCheck, Database }
const ARCH_NODES = [
  { id: 'kb', x: 388, y: 14, w: 244, h: 64, tone: 'sage', icon: 'GitBranch', title: 'Ontology · RDF / OWL', sub: 'classes · aliases · properties' },
  { id: 'cam', x: 18, y: 248, w: 150, h: 92, tone: 'blue', icon: 'Cpu', title: 'RGB-D Camera', sub: 'RGB + depth stream' },
  { id: 'yolo', x: 208, y: 212, w: 172, h: 62, tone: 'blue', icon: 'ScanLine', title: 'YOLO', sub: 'boxes · classes' },
  { id: 'dino', x: 208, y: 288, w: 172, h: 62, tone: 'rust', icon: 'ScanLine', title: 'Grounding DINO', sub: 'prompts · regions' },
  { id: 'gate', x: 418, y: 246, w: 184, h: 92, tone: 'rust', icon: 'Filter', title: 'Dimension gate + common sense', sub: 'physical plausibility filter' },
  { id: 'ground', x: 646, y: 246, w: 184, h: 92, tone: 'sage', icon: 'GitBranch', title: 'Ontology grounding', sub: 'label \u2192 named concept' },
  { id: 'map', x: 646, y: 428, w: 184, h: 92, tone: 'blue', icon: 'Map', title: 'Semantic map', sub: 'landmarks · location · time' },
  { id: 'reason', x: 418, y: 428, w: 184, h: 92, tone: 'rust', icon: 'BrainCircuit', title: 'Common-sense reasoning', sub: 'rules over combined context' },
  { id: 'event', x: 150, y: 428, w: 214, h: 92, tone: 'sage', icon: 'ShieldCheck', title: 'Safety event', sub: 'Exit obstructed + evidence' },
  { id: 'store', x: 656, y: 590, w: 234, h: 62, tone: 'blue', icon: 'Database', title: 'Persistent store', sub: 'SQLite world_map.db \u2194 RDF export' },
]
const ARCH_EDGES = [
  { p: [[168, 294], [208, 243]], label: 'frames', lp: [176, 256] },
  { p: [[168, 294], [208, 319]] },
  { p: [[380, 243], [418, 276]], label: 'detections', lp: [384, 248] },
  { p: [[380, 319], [418, 312]] },
  { p: [[602, 292], [646, 292]], label: 'typed objects', lp: [560, 282] },
  { p: [[738, 338], [738, 428]], label: 'grounded classes', lp: [744, 386] },
  { p: [[646, 474], [602, 474]], label: 'world state', lp: [556, 464] },
  { p: [[418, 474], [364, 474]], label: 'decision', lp: [360, 464] },
  { p: [[558, 78], [738, 246]], label: 'ontology', dashed: true, lp: [648, 150] },
  { p: [[632, 46], [902, 46], [902, 396], [510, 396], [510, 428]], label: 'concepts', dashed: true, lp: [842, 214] },
  { p: [[760, 590], [760, 520]], label: 'write', dashed: true, lp: [766, 558] },
  { p: [[656, 602], [510, 602], [510, 520]], label: 'recall', dashed: true, lp: [556, 594] },
  { p: [[690, 428], [690, 386], [510, 386], [510, 338]], label: 'spatial context', dashed: true, lp: [582, 378] },
]
const ARCH_FLOW = [
  ['RGB-D Camera', 'RGB + depth stream'],
  ['YOLO + Grounding DINO', 'detections + prompts'],
  ['Dimension gate + common sense', 'physical plausibility filter'],
  ['Ontology grounding (RDF/OWL)', 'label \u2192 named concept'],
  ['Semantic map', 'landmarks · location · time'],
  ['Common-sense reasoning', 'rules over combined context'],
  ['Safety event', 'Exit obstructed + evidence'],
]

function ArchDiagram() {
  return <><div className="arch-canvas"><svg className="arch-edges" viewBox={`0 0 ${ARCH_W} ${ARCH_H}`} preserveAspectRatio="none" aria-hidden="true"><defs><marker id="archArrow" markerWidth="9" markerHeight="9" refX="6.4" refY="3" orient="auto-start-reverse"><path d="M0,0 L6.4,3 L0,6 Z" fill="#5f6f67" /></marker></defs>{ARCH_EDGES.map((e, i) => { const pts = e.p.map(p => p.join(',')).join(' '); const [lx, ly] = e.lp || e.p[Math.floor(e.p.length / 2)]; return <g key={i}><polyline points={pts} className={e.dashed ? 'arch-edge dashed' : 'arch-edge'} markerEnd="url(#archArrow)" />{e.label && <text className="arch-edge-label" x={lx} y={ly}>{e.label}</text>}</g> })}</svg>{ARCH_NODES.map(n => { const Icon = ARCH_ICONS[n.icon]; return <div key={n.id} className={`arch-node ${n.tone}`} style={{ left: `${n.x / ARCH_W * 100}%`, top: `${n.y / ARCH_H * 100}%`, width: `${n.w / ARCH_W * 100}%`, height: `${n.h / ARCH_H * 100}%` }}><span className="arch-node-icon"><Icon size={17} /></span><div><strong>{n.title}</strong><small>{n.sub}</small></div></div> })}<span className="arch-band band-top">shared knowledge</span><span className="arch-band band-bottom">persistent memory</span></div><ol className="arch-mobile">{ARCH_FLOW.map((s, i) => <li key={i}><span>{String(i + 1).padStart(2, '0')}</span><div><strong>{s[0]}</strong><small>{s[1]}</small></div></li>)}</ol></>
}

function ArchitecturePage() {
  return <PageFrame eyebrow="System architecture / 02" title={<>A connected view of <em>understanding.</em></>} intro="The framework combines YOLO and Grounding DINO detection with physical constraints, ontology grounding, semantic mapping, and common-sense reasoning to build a situated hospital scene."><section className="architecture-page section-wrap"><div className="architecture-legend"><span><i className="legend-dot dot-blue" /> visual evidence</span><span><i className="legend-dot dot-rust" /> constraints / reasoning</span><span><i className="legend-dot dot-sage" /> semantic world model</span></div><ArchDiagram /><div className="architecture-explanation"><article><Database size={22} /><div><h3>Two complementary detectors</h3><p><strong>YOLO</strong> provides efficient trained object detection. <strong>Grounding DINO</strong> adds open-vocabulary, text-guided grounding. Their outputs enter the same physical and semantic reasoning path.</p></div></article><article><ShieldCheck size={22} /><div><h3>From pixels to a traceable event</h3><p>A state such as <strong>Exit obstructed</strong> can be followed through the door and trolley detections, dimension checks, ontology classes, map location, and common-sense rule.</p></div></article></div><div className="architecture-example"><div className="example-label">Trace example</div><div className="trace-line"><span>YOLO / DINO</span><b>→</b><span>door + trolley</span><b>+</b><span>ontology + map</span><b>→</b><strong>exit obstructed</strong></div><p>Detection is the entry point; grounding, memory, and reasoning are what make the observation useful.</p></div></section></PageFrame>
}

function OntologyBranch({ node, childrenByParent, onSelect, selectedId }) {
  const children = childrenByParent[node.id] || []
  return <div className="ontology-branch"><button className={selectedId === node.id ? 'tree-node selected' : 'tree-node'} onClick={() => onSelect(node)}><span>{node.label}</span><small>{node.id}</small></button>{children.length > 0 && <div className="tree-children">{children.map(child => <OntologyBranch key={child.id} node={child} childrenByParent={childrenByParent} onSelect={onSelect} selectedId={selectedId} />)}</div>}</div>
}

function OntologyPage() {
  const [data, setData] = React.useState(null)
  const [query, setQuery] = React.useState('')
  const [selected, setSelected] = React.useState(null)
  React.useEffect(() => { fetch(`${PUBLIC_BASE}data/ontology.json`).then(response => response.json()).then(payload => { setData(payload); setSelected(payload.classes.find(item => item.id === 'door') || payload.classes[0]) }) }, [])
  const classes = data?.classes || []
  const filtered = query ? classes.filter(item => `${item.id} ${item.label}`.toLowerCase().includes(query.toLowerCase())) : []
  const classIds = new Set(classes.map(item => item.id))
  const childrenByParent = classes.reduce((groups, item) => { item.parents.filter(parent => classIds.has(parent)).forEach(parent => { groups[parent] = [...(groups[parent] || []), item] }); return groups }, {})
  const roots = classes.filter(item => item.parents.length === 0 || item.parents.every(parent => !classIds.has(parent)))
  return <PageFrame eyebrow="Ontology explorer / 03" title={<>Meaning, made <em>explicit.</em></>} intro="An interactive view of the RDF/OWL knowledge model. Select a class to inspect its identifier and parent relationships, then search the graph by detector label or ontology name."><section className="ontology-page section-wrap"><div className="ontology-stats"><div><strong>{data?.classCount || '—'}</strong><span>classes from RDF</span></div><div><strong>{data?.propertyCount || '—'}</strong><span>properties from RDF</span></div><div><strong>RDF/XML</strong><span>authoritative source</span></div><label className="ontology-search">Search classes<input value={query} onChange={event => setQuery(event.target.value)} placeholder="door, hazard, wheelchair..." /></label></div>{query && <div className="search-results">{filtered.slice(0, 12).map(item => <button key={item.id} onClick={() => { setSelected(item); setQuery('') }}>{item.label}<small>{item.id}</small></button>)}{filtered.length === 0 && <span>No classes match this search.</span>}</div>}<div className="ontology-workspace"><div className="ontology-tree"><div className="workspace-heading"><div><p className="eyebrow">Class hierarchy</p><h2>The RDF graph at a glance.</h2></div><span>click any node</span></div>{data ? <div className="tree-roots">{roots.map(root => <OntologyBranch key={root.id} node={root} childrenByParent={childrenByParent} onSelect={setSelected} selectedId={selected?.id} />)}</div> : <div className="loading-state">Loading ontology.rdf graph...</div>}</div><aside className="class-inspector"><p className="eyebrow">Selected class</p>{selected ? <><h3>{selected.label}</h3><code>{selected.id}</code><div className="inspector-row"><span>URI</span><strong>{selected.uri}</strong></div><div className="inspector-row"><span>Parent classes</span><strong>{selected.parents.length ? selected.parents.join(', ') : 'Top-level concept'}</strong></div><div className="inspector-row"><span>Child classes</span><strong>{(childrenByParent[selected.id] || []).length}</strong></div></> : <p>Choose a node to inspect its RDF identity.</p>}</aside></div><div className="property-table"><div className="workspace-heading"><div><p className="eyebrow">RDF relationships</p><h2>Properties carried by the model.</h2></div></div><div className="property-list">{(data?.properties || []).map(property => <div className="property-row" key={`${property.type}-${property.id}`}><strong>{property.label}</strong><span>{property.type}</span><small>{property.domain || 'Any subject'} → {property.range || 'Any object'}</small></div>)}</div></div></section></PageFrame>
}

function RobotPage() {
  return <PageFrame eyebrow="Platform record / 04" title={<>A robot that can <em>remember.</em></>} intro="The platform brings RGB-D sensing, trained detection, ontology grounding, and persistent semantic mapping together on a mobile robot." ><section className="section-wrap page-content"><div className="robot-layout"><figure className="robot-portrait robot-photo"><img src={`${PUBLIC_BASE}media/kobuki-robot.png`} alt="Kobuki mobile robot platform with RGB-D sensing and onboard compute" /><figcaption>Kobuki mobile robot platform with RGB-D sensing and onboard compute</figcaption></figure><div className="spec-list"><div className="spec-row"><span>Perception</span><strong>YOLO object detection<br /><small>RGB-D image streams</small></strong></div><div className="spec-row"><span>World model</span><strong>Persistent semantic map<br /><small>SQLite landmarks · RDF/OWL export</small></strong></div><div className="spec-row"><span>Knowledge</span><strong>Hospital object ontology<br /><small>Aliases · dimensions · relationships</small></strong></div><div className="spec-row"><span>Output</span><strong>Interpretable safety events<br /><small>Door obstruction · anomalies · spatial rules</small></strong></div></div></div><div className="detail-grid robot-detail"><article><p className="eyebrow">System relationship</p><h2>Hardware becomes useful when it can <em>place</em> what it sees.</h2></article><article><p>RGB-D sensing gives the system depth and geometry. The semantic map retains landmarks over time. Ontology resolution gives labels a shared vocabulary that can travel from detector output into RDF, viewers, and rules.</p><a className="button button-dark" href="#/ontology">Explore the ontology <ArrowUpRight size={17} /></a></article></div></section></PageFrame>
}

function AboutPage() {
  return <PageFrame eyebrow="Project record / 05" title={<>Built in the open, <em>updated with evidence.</em></>} intro="A public research record for the Cognitive Recognition Framework by Krishna Pavani Munta."><section className="section-wrap page-content"><div className="detail-grid"><article><p className="eyebrow">Project scope</p><h2>Perception, knowledge, and safety in one research story.</h2><p>This website brings together experiments, models, robot details, demonstrations, ontology assets, and semantic-map outputs as the framework develops.</p></article><article className="quote-panel"><span className="quote-mark">©</span><p>Krishna Pavani Munta</p><small>Research author / 2026</small></article></div><div className="update-timeline"><div><span>01</span><strong>Build</strong><p>Train and evaluate perception models.</p></div><div><span>02</span><strong>Ground</strong><p>Connect detections to ontology classes.</p></div><div><span>03</span><strong>Publish</strong><p>Release approved results and demonstrations.</p></div></div></section></PageFrame>
}

function App() {
  const [menuOpen, setMenuOpen] = React.useState(false)
  const [route, setRoute] = React.useState(window.location.hash || '#/')
  React.useEffect(() => { const updateRoute = () => setRoute(window.location.hash || '#/'); window.addEventListener('hashchange', updateRoute); return () => window.removeEventListener('hashchange', updateRoute) }, [])

  if (route === '#/research') return <ResearchPage />
  if (route === '#/architecture') return <ArchitecturePage />
  if (route === '#/demonstrations') return <DemonstrationsPage />
  if (route === '#/ontology') return <OntologyPage />
  if (route === '#/robot') return <RobotPage />
  if (route === '#/about') return <AboutPage />

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Cognitive Recognition Framework home">
          <span className="brand-mark">CRF</span>
          <span><strong>Cognitive Recognition</strong><small>Framework</small></span>
        </a>
        <button className="menu-button" type="button" aria-label="Toggle navigation" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X size={20} /> : <Menu size={20} />}</button>
        <nav className={menuOpen ? 'nav-links open' : 'nav-links'}>
          <a href="#/research" onClick={() => setMenuOpen(false)}>Research</a>
          <a href="#/architecture" onClick={() => setMenuOpen(false)}>Architecture</a>
          <a href="#/demonstrations" onClick={() => setMenuOpen(false)}>Demonstrations</a>
          <a href="#/ontology" onClick={() => setMenuOpen(false)}>Ontology</a>
          <a href="#/robot" onClick={() => setMenuOpen(false)}>Robot</a>
          <a href="#/about" onClick={() => setMenuOpen(false)}>About</a>
        </nav>
        <a className="header-link" href="https://github.com/KrishnaPavaniMunta" target="_blank" rel="noreferrer" aria-label="Krishna Pavani Munta on GitHub"><SiGithub size={18} /> GitHub</a>
      </header>

      <main id="top">
        <section className="hero section-wrap">
          <div className="hero-copy reveal">
            <div className="kicker"><span className="status-dot" /> Research in progress <span className="rule" /> 2026</div>
            <h1>Giving hospital robots a more <em>thoughtful</em> view of the world.</h1>
            <p className="hero-lede">A cognitive recognition framework that joins visual perception, physical knowledge, spatial memory, and safety reasoning for hospital environments.</p>
            <div className="hero-actions"><a className="button button-dark" href="#research">Explore the framework <ArrowUpRight size={17} /></a><a className="text-link" href="#demonstrations">Watch demonstrations <span>↓</span></a></div>
          </div>
          <div className="hero-aside reveal reveal-delay">
            <div className="hero-diagram" aria-label="Framework diagram">
              <div className="diagram-ring ring-one" /><div className="diagram-ring ring-two" />
              <div className="diagram-core"><CircleDot size={24} /><span>semantic<br />understanding</span></div>
              <span className="diagram-label label-top">RGB-D<br />perception</span><span className="diagram-label label-right">ontology<br />grounding</span><span className="diagram-label label-bottom">safety<br />reasoning</span><span className="diagram-label label-left">spatial<br />memory</span>
            </div>
            <p className="aside-caption">From pixels to situated knowledge</p>
          </div>
        </section>

        <section className="signal-strip"><div className="section-wrap signal-inner"><span>Open University · Knowledge Media Institute</span><span className="signal-line" /><span>Object detection · RGB-D mapping · Ontology</span></div></section>

        <section className="section-wrap research-section" id="research">
          <div className="section-heading"><span className="section-number">01 / 04</span><div><p className="eyebrow">The research</p><h2>Recognition needs<br /><em>context.</em></h2></div><p className="section-intro">The project explores how a robot can move beyond naming objects. By combining learned perception with explicit knowledge and spatial persistence, the system can ask what an observation means in its environment.</p></div>
          <div className="steps-grid">{systemSteps.map(([number, title, copy]) => <article className="step" key={number}><span className="step-number">{number}</span><h3>{title}</h3><p>{copy}</p></article>)}</div>
        </section>

        <section className="dark-band" id="demonstrations"><div className="section-wrap"><div className="section-heading light"><span className="section-number">02 / 04</span><div><p className="eyebrow">Working examples</p><h2>From detection<br />to <em>decision.</em></h2></div><p className="section-intro">These early demonstrations show how the framework constrains perception with what is physically plausible and operationally important.</p></div><div className="demo-grid">{demonstrations.map((demo) => <article className="demo-card" key={demo.id}><div className="video-frame"><video controls preload="metadata" src={demo.video}><track kind="captions" /></video><span className="video-index">{demo.eyebrow}</span></div><div className="demo-copy"><p className="eyebrow">{demo.eyebrow}</p><h3>{demo.title}</h3><p>{demo.description}</p><div className="tag-list">{demo.tags.map(tag => <span key={tag}>{tag}</span>)}</div></div></article>)}</div></div></section>

        <section className="ontology-section" id="ontology"><div className="section-wrap"><div className="section-heading"><span className="section-number">03 / 04</span><div><p className="eyebrow">The ontology</p><h2>Meaning, made<br /><em>explicit.</em></h2></div><p className="section-intro">The ontology is the bridge between a detector label and a useful understanding of the hospital. It gives every observation a class, physical expectations, a place in the map, and relationships to other things.</p></div><div className="ontology-layout"><div className="ontology-copy"><div className="ontology-note"><span className="note-marker">A</span><div><strong>Why this layer matters</strong><p>Two objects can look similar but mean very different things in context. Ontology grounding makes those distinctions inspectable instead of hidden inside a model.</p></div></div><div className="ontology-note"><span className="note-marker">B</span><div><strong>What the system carries forward</strong><p>Aliases, dimensions, spatial relations, confidence, and time become structured knowledge that downstream rules can reason over.</p></div></div></div><div className="ontology-map" aria-label="Visual ontology connection map"><div className="map-column map-input"><span className="map-label">Observed</span><div className="map-node node-image"><span className="node-kicker">RGB-D frame</span><strong>door</strong><small>label + depth</small></div><div className="map-node node-object"><span className="node-kicker">detection</span><strong>trolley</strong><small>box + confidence</small></div></div><div className="map-connectors"><i /><i /><i /></div><div className="map-column map-knowledge"><span className="map-label">Understood</span><div className="map-node node-ontology"><span className="node-kicker">ontology class</span><strong>HospitalDoor</strong><small>alias · dimensions</small></div><div className="map-node node-spatial"><span className="node-kicker">spatial instance</span><strong>near(exit_01)</strong><small>map landmark</small></div></div><div className="map-arrow">→</div><div className="map-column map-action"><span className="map-label">Reasoned</span><div className="map-node node-event"><span className="node-kicker">safety rule</span><strong>Exit obstructed</strong><small>door + trolley + distance</small></div><div className="event-chip"><span /> interpretable event</div></div></div></div></div></section>

        <section className="section-wrap robot-section" id="robot"><div className="section-heading"><span className="section-number">04 / 04</span><div><p className="eyebrow">The platform</p><h2>A robot that can<br /><em>remember.</em></h2></div><p className="section-intro">The software stack is built around a mobile robot carrying RGB-D sensing, a trained object detector, and a persistent semantic representation of its surroundings.</p></div><div className="robot-layout"><div className="robot-portrait"><div className="robot-grid" /><div className="robot-icon"><Cpu size={54} strokeWidth={1.2} /><span>RGB-D<br />mobile platform</span></div><span className="portrait-label">SYSTEM / 01</span></div><div className="spec-list"><div className="spec-row"><span>Perception</span><strong>YOLO object detection<br /><small>RGB-D image streams</small></strong></div><div className="spec-row"><span>World model</span><strong>Persistent semantic map<br /><small>SQLite landmarks · RDF/OWL export</small></strong></div><div className="spec-row"><span>Knowledge</span><strong>Hospital object ontology<br /><small>Aliases · dimensions · relationships</small></strong></div><div className="spec-row"><span>Output</span><strong>Interpretable safety events<br /><small>Door obstruction · anomalies · spatial rules</small></strong></div></div></div></section>

        <section className="paper-band" id="about"><div className="section-wrap paper-layout"><div><p className="eyebrow">A living research record</p><h2>Built in the open,<br /><em>updated with evidence.</em></h2></div><div className="paper-copy"><p>This site will grow alongside the framework: new experiments, model versions, videos, maps, and findings can be added without losing the story of how they were produced.</p><div className="update-note"><Check size={17} /><span><strong>Update cadence</strong><br />Curated releases from the research repository</span></div><a className="button button-light" href="#research">Read the project record <ExternalLink size={16} /></a></div></div></section>
      </main>

      <footer className="site-footer"><div className="footer-top"><div className="footer-brand"><span className="brand-mark">CRF</span><div><strong>Cognitive Recognition Framework</strong><small>Hospital robotics research</small></div></div><div className="institutional-marks"><InstitutionalMark type="ou-mark" /><InstitutionalMark type="kmi-mark" /><InstitutionalMark type="resilient-mark" /></div></div><div className="footer-bottom"><span>© 2026 Krishna Pavani Munta. All rights reserved.</span><span>Research website · Public project record</span><a href="https://github.com/KrishnaPavaniMunta" target="_blank" rel="noreferrer" aria-label="Krishna Pavani Munta on GitHub"><SiGithub size={17} /></a></div><p className="viewing-note">Best viewed on a laptop or computer screen.</p></footer>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
