# Cognitive Recognition Framework website

A public-facing research website for the hospital cognitive-recognition framework.

## Run locally

```powershell
cd website
npm install
npm run dev
```

Open the local URL printed by Vite.

## Build for deployment

```powershell
npm run build
npm run preview
```

The site is a self-contained Vite app. Demonstration videos live in `public/media`. Replace the text-based institutional marks in `src/main.jsx` with authorized Open University and KMi logo assets before public publication. Keep research footage and personal data reviewed for publication permissions.

## Pages

- `#/` - project overview
- `#/research` - detailed research record
- `#/architecture` - visual architecture and evidence flow
- `#/demonstrations` - video demonstrations and reasoning chain
- `#/ontology` - searchable RDF class hierarchy, selected-class inspector, and property list
- `#/robot` - platform and system details
- `#/about` - project record and update timeline

## Refresh the ontology viewer

The ontology page is generated from the repository's authoritative RDF/XML file. After changing `01_codebase/09_ontology/ontology.rdf`, run:

```powershell
npm run ontology:export
npm run build
```

This regenerates `public/data/ontology.json`, which the ontology page loads at runtime.

## Updating the research record

Add approved demonstration videos to `public/media`, then update the `demonstrations` list in `src/main.jsx`. Add new metrics, model versions, or findings to the relevant section and publish the generated `dist` folder through a public host such as GitHub Pages, Netlify, or Vercel. A scheduled CI workflow can run the same build after approved repository changes are merged.
