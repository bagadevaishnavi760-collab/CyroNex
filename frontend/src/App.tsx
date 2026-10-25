import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  Activity,
  Atom,
  ArrowDown,
  ArrowRight,
  BarChart3,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Database,
  FlaskConical,
  History,
  LayoutDashboard,
  Microscope,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  Thermometer,
  Trash2,
} from 'lucide-react';
import {
  candidateCompleteness,
  orderedFeaturePayload,
  outOfRangeFeatures,
  rankCandidates,
  type CandidateExplanation,
  type CandidateProfile,
  type FeatureRange,
} from './discovery';
import './platform.css';

const API = (import.meta.env.VITE_API_BASE_URL || 'https://cyronex.onrender.com').replace(/\/$/, '');
const PAGE_SIZE = 20;
const REQUEST_TIMEOUT_MS = 60_000;
const DISCLAIMER =
  'Model estimates rank candidates computationally; they do not establish chemical feasibility or experimental superconductivity.';
const PREDICTION_DISCLAIMER =
  'This is a machine-learning prediction based on the supplied material features. It is an estimate and does not constitute experimental confirmation.';

type Page = 'discovery' | 'comparison' | 'dataset' | 'insights' | 'history';
type FeatureGroup = { name: string; features: string[] };
type FeatureConfig = { features: string[]; groups: FeatureGroup[] };
type DatasetRecord = { index: number; values: Record<string, number> | null; target: number | null };
type DatasetSummary = {
  rows: number;
  features: number;
  missing_values: number;
  target_stats: { min: number; max: number; mean: number; median: number };
  feature_stats: FeatureRange[];
};
type ModelInsights = {
  model: string;
  feature_count: number;
  importance: Array<{ feature: string; importance: number }>;
};
type PredictionHistory = {
  id: string;
  at: string;
  candidateName: string;
  prediction: number;
  sourceIndex: number | null;
  values?: Record<string, number>;
};

const navigation = [
  { id: 'discovery' as const, label: 'Material Discovery', icon: FlaskConical },
  { id: 'comparison' as const, label: 'Candidate Comparison', icon: BarChart3 },
  { id: 'dataset' as const, label: 'Dataset Explorer', icon: Database },
  { id: 'insights' as const, label: 'Model Insights', icon: Microscope },
  { id: 'history' as const, label: 'History', icon: History },
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

async function requestJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${API}${path}`, {
      ...init,
      signal: controller.signal,
    });
    const payload: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const message =
        isRecord(payload) && typeof payload.error === 'string'
          ? payload.error
          : `CryoNex API returned an error (${response.status}).`;
      throw new Error(message);
    }
    if (payload === null) throw new Error('The API returned an unreadable response.');
    return payload as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('The API took too long to respond. Please try again.');
    }
    if (error instanceof TypeError) {
      throw new Error('Could not reach the CryoNex API. Check the connection and try again.');
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function readFeatureConfig(response: unknown): FeatureConfig {
  if (
    !isRecord(response) ||
    !Array.isArray(response.features) ||
    response.features.length !== 81 ||
    response.features.some((feature) => typeof feature !== 'string') ||
    new Set(response.features).size !== 81 ||
    response.features.includes('critical_temp') ||
    !Array.isArray(response.groups)
  ) {
    throw new Error('The API did not return the model’s complete 81-feature schema.');
  }
  const features = response.features as string[];
  const groups = response.groups;
  if (
    groups.some(
      (group) =>
        !isRecord(group) ||
        typeof group.name !== 'string' ||
        !Array.isArray(group.features) ||
        group.features.some((feature) => typeof feature !== 'string'),
    )
  ) {
    throw new Error('The API returned invalid feature group definitions.');
  }
  const typedGroups = groups as FeatureGroup[];
  const groupedFeatures = typedGroups.flatMap((group) => group.features);
  if (
    groupedFeatures.length !== features.length ||
    new Set(groupedFeatures).size !== features.length ||
    groupedFeatures.some((feature, index) => feature !== features[index])
  ) {
    throw new Error('The feature groups do not match the trained model feature order.');
  }
  return { features, groups: typedGroups };
}

function readDatasetPage(
  response: unknown,
  features: string[],
): { records: DatasetRecord[]; page: number; total: number } {
  if (
    !isRecord(response) ||
    !Array.isArray(response.rows) ||
    !Number.isInteger(response.page) ||
    !Number.isInteger(response.page_size) ||
    !Number.isInteger(response.total)
  ) {
    throw new Error('The dataset endpoint returned an unexpected response.');
  }
  const rows = response.rows;
  const page = response.page as number;
  const pageSize = response.page_size as number;
  const total = response.total as number;
  if (pageSize < 1 || total < 0 || rows.some((row) => !isRecord(row))) {
    throw new Error('The dataset endpoint returned invalid page data.');
  }
  return {
    page,
    total,
    records: (rows as Record<string, unknown>[]).map((row, rowIndex) => {
      const values = Object.fromEntries(
        features.map((feature) => [feature, row[feature]]),
      );
      const complete = features.every((feature) => isFiniteNumber(values[feature]));
      return {
        index: (page - 1) * pageSize + rowIndex,
        values: complete ? (values as Record<string, number>) : null,
        target: isFiniteNumber(row.critical_temp) ? row.critical_temp : null,
      };
    }),
  };
}

function readSample(response: unknown, features: string[]): DatasetRecord {
  if (
    !isRecord(response) ||
    !Number.isInteger(response.index) ||
    !isRecord(response.features) ||
    Object.keys(response.features).length !== features.length
  ) {
    throw new Error('The dataset sample is incomplete and cannot be used as a template.');
  }
  const values: Record<string, number> = {};
  for (const feature of features) {
    const value = response.features[feature];
    if (!isFiniteNumber(value)) {
      throw new Error(`Dataset template is missing a valid “${feature}” value.`);
    }
    values[feature] = value;
  }
  return {
    index: response.index as number,
    values,
    target: isFiniteNumber(response.critical_temp) ? response.critical_temp : null,
  };
}

function readPrediction(response: unknown): number {
  if (
    !isRecord(response) ||
    !isFiniteNumber(response.predicted_critical_temperature_K)
  ) {
    throw new Error('The API did not return a valid predicted critical temperature.');
  }
  return response.predicted_critical_temperature_K;
}

function readExplanation(response: unknown, features: string[]): CandidateExplanation {
  if (
    !isRecord(response) ||
    typeof response.method !== 'string' ||
    !isFiniteNumber(response.prediction) ||
    !isFiniteNumber(response.base_value) ||
    !Array.isArray(response.contributions)
  ) {
    throw new Error('A valid per-candidate model explanation was not returned.');
  }
  const rawContributions = response.contributions;
  if (rawContributions.length !== features.length) {
    throw new Error('The explanation does not match the ordered 81-feature model input.');
  }
  const parsed: CandidateExplanation['contributions'] = [];
  for (let index = 0; index < rawContributions.length; index += 1) {
    const item: unknown = rawContributions[index];
    if (
      !isRecord(item) ||
      item.feature !== features[index] ||
      !isFiniteNumber(item.value) ||
      !isFiniteNumber(item.contribution)
    ) {
      throw new Error('The explanation does not match the ordered 81-feature model input.');
    }
    parsed.push({
      feature: item.feature,
      value: item.value,
      contribution: item.contribution,
    });
  }
  const prediction = response.prediction;
  const baseValue = response.base_value;
  if (!isFiniteNumber(prediction) || !isFiniteNumber(baseValue)) {
    throw new Error('The model explanation returned invalid values.');
  }
  const sum = parsed.reduce((total, item) => total + item.contribution, baseValue);
  if (Math.abs(sum - prediction) > Math.max(1e-4, Math.abs(prediction) * 1e-4)) {
    throw new Error('The model explanation did not reconcile with its prediction.');
  }
  return {
    method: response.method,
    prediction,
    base_value: baseValue,
    contributions: parsed,
  };
}

function newCandidate(number: number): CandidateProfile {
  return {
    id: crypto.randomUUID(),
    name: `Candidate ${number}`,
    sourceIndex: null,
    values: {},
    prediction: null,
  };
}

function readCandidates(): CandidateProfile[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem('cryonex-candidates') || '[]');
    if (!Array.isArray(raw)) return [newCandidate(1)];
    const candidates = raw.filter(
      (item): item is CandidateProfile =>
        isRecord(item) &&
        typeof item.id === 'string' &&
        typeof item.name === 'string' &&
        (item.sourceIndex === null ||
          (Number.isInteger(item.sourceIndex) && Number(item.sourceIndex) >= 0)) &&
        isRecord(item.values) &&
        Object.values(item.values).every(
          (value) => typeof value === 'string' || isFiniteNumber(value),
        ) &&
        (item.prediction === null || isFiniteNumber(item.prediction)),
    );
    return candidates.length ? candidates : [newCandidate(1)];
  } catch {
    return [newCandidate(1)];
  }
}

function readHistory(): PredictionHistory[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem('cryonex-history') || '[]');
    if (!Array.isArray(raw)) return [];
    return raw.flatMap((item) => {
      if (!isRecord(item) || typeof item.id !== 'string' || !isFiniteNumber(item.prediction)) {
        return [];
      }
      const values = isRecord(item.values)
        ? item.values
        : isRecord(item.features)
          ? item.features
          : undefined;
      const numericValues =
        values && Object.values(values).every(isFiniteNumber)
          ? (values as Record<string, number>)
          : undefined;
      return [{
        id: item.id,
        at: typeof item.at === 'string' ? item.at : new Date().toISOString(),
        candidateName:
          typeof item.candidateName === 'string'
            ? item.candidateName
            : typeof item.recordIndex === 'number'
              ? `Dataset candidate #${item.recordIndex + 1}`
              : 'Saved estimate',
        prediction: item.prediction,
        sourceIndex: Number.isInteger(item.sourceIndex)
          ? (item.sourceIndex as number)
          : Number.isInteger(item.recordIndex)
            ? (item.recordIndex as number)
            : null,
        values: numericValues,
      }];
    });
  } catch {
    return [];
  }
}

const formatNumber = (value: number) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value);
const displayName = (feature: string) =>
  feature.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const displayGroupName = (name: string) =>
  ({
    Composition: 'Element count',
    Fie: 'First ionization energy (FIE)',
    Electronaffinity: 'Electron affinity',
    Fusionheat: 'Heat of fusion',
    Thermalconductivity: 'Thermal conductivity',
  })[name] ?? name;
const describeError = (error: unknown, fallback: string) =>
  error instanceof Error ? error.message : fallback;

export default function App() {
  const [page, setPage] = useState<Page>('discovery');
  const [configRetry, setConfigRetry] = useState(0);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [groups, setGroups] = useState<FeatureGroup[]>([]);
  const [candidates, setCandidates] = useState<CandidateProfile[]>(readCandidates);
  const [selectedCandidateId, setSelectedCandidateId] = useState('');
  const [templateIndex, setTemplateIndex] = useState('0');
  const [loadingFeatures, setLoadingFeatures] = useState(true);
  const [loadingTemplate, setLoadingTemplate] = useState(false);
  const [runningIds, setRunningIds] = useState<string[]>([]);
  const [explainingId, setExplainingId] = useState('');
  const [apiError, setApiError] = useState('');
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [insights, setInsights] = useState<ModelInsights | null>(null);
  const [records, setRecords] = useState<DatasetRecord[]>([]);
  const [recordsPage, setRecordsPage] = useState(1);
  const [recordsTotal, setRecordsTotal] = useState(0);
  const [recordsQuery, setRecordsQuery] = useState('');
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [history, setHistory] = useState<PredictionHistory[]>(readHistory);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [globalSearch, setGlobalSearch] = useState('');
  const [explainCandidateId, setExplainCandidateId] = useState('');
  const [explanationError, setExplanationError] = useState('');

  const selectedCandidate =
    candidates.find((candidate) => candidate.id === selectedCandidateId) ?? candidates[0];
  const featureRanges = summary?.feature_stats ?? [];
  const featureRangeMap = useMemo(
    () => new Map(featureRanges.map((range) => [range.feature, range])),
    [featureRanges],
  );
  const rankedCandidates = useMemo(() => rankCandidates(candidates), [candidates]);
  const groupsFiltered = useMemo(
    () =>
      groups
        .map((group) => ({
          ...group,
          features: group.features.filter((feature) =>
            feature.toLowerCase().includes(globalSearch.toLowerCase()),
          ),
        }))
        .filter((group) => group.features.length > 0),
    [groups, globalSearch],
  );
  const explainCandidate =
    candidates.find((candidate) => candidate.id === explainCandidateId) ??
    candidates.find((candidate) => candidate.explanation) ??
    candidates[0];

  useEffect(() => {
    if (!selectedCandidateId && candidates[0]) setSelectedCandidateId(candidates[0].id);
  }, [candidates, selectedCandidateId]);

  useEffect(() => {
    localStorage.setItem('cryonex-candidates', JSON.stringify(candidates));
  }, [candidates]);

  useEffect(() => {
    localStorage.setItem('cryonex-history', JSON.stringify(history.slice(0, 100)));
  }, [history]);

  useEffect(() => {
    let active = true;
    setLoadingFeatures(true);
    setApiError('');
    void requestJSON<unknown>('/features')
      .then((response) => {
        const config = readFeatureConfig(response);
        if (!active) return;
        setFeatureNames(config.features);
        setGroups(config.groups);
        setLoadingFeatures(false);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setApiError(describeError(error, 'Could not load the model feature schema.'));
        setLoadingFeatures(false);
      });
    void requestJSON<DatasetSummary>('/dataset/summary')
      .then((data) => active && setSummary(data))
      .catch(() => undefined);
    void requestJSON<ModelInsights>('/model/insights')
      .then((data) => active && setInsights(data))
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [configRetry]);

  async function loadRecords(pageNumber: number, query = recordsQuery) {
    if (featureNames.length !== 81) {
      setApiError('Load the complete 81-feature schema before browsing dataset rows.');
      return;
    }
    setRecordsLoading(true);
    setApiError('');
    try {
      const search = new URLSearchParams({
        page: String(pageNumber),
        page_size: String(PAGE_SIZE),
        q: query,
      });
      const response = await requestJSON<unknown>(`/dataset/records?${search.toString()}`);
      const parsed = readDatasetPage(response, featureNames);
      setRecords(parsed.records);
      setRecordsPage(parsed.page);
      setRecordsTotal(parsed.total);
    } catch (error) {
      setApiError(describeError(error, 'Dataset records could not be loaded.'));
    } finally {
      setRecordsLoading(false);
    }
  }

  useEffect(() => {
    if (page === 'dataset' && featureNames.length === 81) {
      void loadRecords(recordsPage, recordsQuery);
    }
  }, [page, featureNames.length]);

  function updateCandidate(
    id: string,
    updater: (candidate: CandidateProfile) => CandidateProfile,
  ) {
    setCandidates((current) => current.map((candidate) =>
      candidate.id === id ? updater(candidate) : candidate,
    ));
  }

  function addCandidate() {
    const candidate = newCandidate(candidates.length + 1);
    setCandidates((current) => [...current, candidate]);
    setSelectedCandidateId(candidate.id);
    setApiError('');
    setPage('discovery');
  }

  function updateCandidateName(id: string, name: string) {
    updateCandidate(id, (candidate) => ({ ...candidate, name }));
  }

  function updateCandidateValue(id: string, feature: string, value: string) {
    updateCandidate(id, (candidate) => ({
      ...candidate,
      values: { ...candidate.values, [feature]: value },
      prediction: null,
      explanation: undefined,
      error: undefined,
    }));
  }

  async function loadTemplate(indexValue: string | number = templateIndex, candidateId = selectedCandidate?.id) {
    const index = typeof indexValue === 'number' ? indexValue : Number(indexValue);
    if (
      !candidateId ||
      (typeof indexValue === 'string' && indexValue.trim() === '') ||
      !Number.isInteger(index) ||
      index < 0
    ) {
      setApiError('Enter a valid non-negative dataset row number to load a template.');
      return;
    }
    setLoadingTemplate(true);
    setApiError('');
    try {
      const response = await requestJSON<unknown>(`/sample?index=${index}`);
      const record = readSample(response, featureNames);
      if (record.index !== index) {
        throw new Error('The API returned a different dataset row than the selected template.');
      }
      updateCandidate(candidateId, (candidate) => ({
        ...candidate,
        sourceIndex: record.index,
        values: record.values ?? {},
        prediction: null,
        explanation: undefined,
        error: undefined,
      }));
    } catch (error) {
      setApiError(describeError(error, 'A valid dataset template could not be loaded.'));
    } finally {
      setLoadingTemplate(false);
    }
  }

  async function predictCandidate(candidate: CandidateProfile) {
    const completeness = candidateCompleteness(candidate.values, featureNames);
    if (!completeness.complete) {
      const unresolved = [...completeness.missing, ...completeness.invalid];
      updateCandidate(candidate.id, (current) => ({
        ...current,
        error:
          unresolved.length > 0
            ? `Complete or correct ${unresolved.length} feature${unresolved.length === 1 ? '' : 's'} before predicting: ${unresolved.slice(0, 5).join(', ')}${unresolved.length > 5 ? ` and ${unresolved.length - 5} more` : ''}.`
            : 'The model feature schema is unavailable.',
      }));
      return false;
    }
    setRunningIds((current) => current.includes(candidate.id) ? current : [...current, candidate.id]);
    updateCandidate(candidate.id, (current) => ({ ...current, error: undefined }));
    try {
      const payload = orderedFeaturePayload(candidate.values, featureNames);
      const response = await requestJSON<unknown>('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const prediction = readPrediction(response);
      updateCandidate(candidate.id, (current) => ({
        ...current,
        prediction,
        explanation: undefined,
        error: undefined,
      }));
      setHistory((current) => [{
        id: crypto.randomUUID(),
        at: new Date().toISOString(),
        candidateName: candidate.name || 'Unnamed candidate',
        prediction,
        sourceIndex: candidate.sourceIndex,
        values: payload,
      }, ...current].slice(0, 100));
      return true;
    } catch (error) {
      updateCandidate(candidate.id, (current) => ({
        ...current,
        error: describeError(error, 'The model prediction failed. Please try again.'),
      }));
      return false;
    } finally {
      setRunningIds((current) => current.filter((id) => id !== candidate.id));
    }
  }

  async function predictAllCompleteCandidates() {
    for (const candidate of candidates) {
      if (candidateCompleteness(candidate.values, featureNames).complete) {
        await predictCandidate(candidate);
      }
    }
  }

  async function explain(candidate: CandidateProfile) {
    setExplanationError('');
    const completeness = candidateCompleteness(candidate.values, featureNames);
    if (!completeness.complete) {
      setExplanationError('Complete all 81 features before requesting a candidate explanation.');
      return;
    }
    setExplainingId(candidate.id);
    try {
      const payload = orderedFeaturePayload(candidate.values, featureNames);
      const response = await requestJSON<unknown>('/model/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = readExplanation(response, featureNames);
      updateCandidate(candidate.id, (current) => ({
        ...current,
        prediction: result.prediction,
        explanation: result,
        error: undefined,
      }));
      setExplainCandidateId(candidate.id);
    } catch (error) {
      setExplanationError(describeError(error, 'The model explanation is unavailable.'));
      updateCandidate(candidate.id, (current) => ({ ...current, explanation: undefined }));
    } finally {
      setExplainingId('');
    }
  }

  function selectCandidate(id: string) {
    setSelectedCandidateId(id);
    setApiError('');
  }

  function goTo(nextPage: Page) {
    setPage(nextPage);
    setApiError('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function removeCandidate(id: string) {
    const remaining = candidates.filter((candidate) => candidate.id !== id);
    if (!remaining.length) remaining.push(newCandidate(1));
    if (selectedCandidateId === id) setSelectedCandidateId(remaining[0].id);
    setCandidates(remaining);
  }

  const selectedCompleteness = selectedCandidate
    ? candidateCompleteness(selectedCandidate.values, featureNames)
    : { complete: false, supplied: 0, missing: featureNames, invalid: [] };
  const selectedOutOfRange = selectedCandidate
    ? outOfRangeFeatures(selectedCandidate.values, featureRanges)
    : [];
  const observedRangeCount = featureRanges.filter(
    (range) => isFiniteNumber(range.min) && isFiniteNumber(range.max),
  ).length;

  return (
    <div className="app-shell platform-shell">
      <header className="topbar platform-header">
        <button className="brand platform-brand" onClick={() => goTo('discovery')} aria-label="CryoNex Material Discovery">
          <span className="brand-mark"><Atom size={19} /></span>
          <span>CRYO<span className="brand-accent">NEX</span><small>QUANTUM MATERIALS INTELLIGENCE</small></span>
        </button>
        <nav className="desktop-nav platform-nav" aria-label="Main navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={page === id ? 'active' : ''}
              aria-current={page === id ? 'page' : undefined}
              onClick={() => goTo(id)}
            >
              <Icon size={15} />{label}
            </button>
          ))}
        </nav>
        <span className="platform-model-status"><i /> XGBoost model</span>
      </header>

      <main className="platform-main page-width">
        {page === 'discovery' && (
          <>
            <PageHead
              kicker="AI-ASSISTED MATERIAL WORKSPACE"
              title={<>Material <span>Discovery</span></>}
              sub="Build a candidate from an observed dataset template or enter its complete model descriptors. Predict critical temperature without treating a recorded target as an input."
            />
            <div className="discovery-toolbar">
              <div className="candidate-tabs" role="tablist" aria-label="Candidate profiles">
                {candidates.map((candidate) => (
                  <button
                    key={candidate.id}
                    role="tab"
                    aria-selected={candidate.id === selectedCandidate?.id}
                    className={candidate.id === selectedCandidate?.id ? 'selected' : ''}
                    onClick={() => selectCandidate(candidate.id)}
                  >
                    <Atom size={14} />{candidate.name || 'Unnamed candidate'}
                  </button>
                ))}
              </div>
              <button className="button secondary small" onClick={addCandidate}><Plus size={14} /> New candidate</button>
            </div>

            {selectedCandidate ? (
              <div className="discovery-layout">
                <section className="panel candidate-editor">
                  <div className="section-title-row">
                    <span className="step-marker">01</span>
                    <div><span className="eyebrow tiny">CANDIDATE PROFILE</span><h2>Prepare model inputs</h2></div>
                    <button className="icon-button danger-icon" aria-label="Delete candidate" onClick={() => removeCandidate(selectedCandidate.id)} disabled={candidates.length === 1}><Trash2 size={16} /></button>
                  </div>
                  <label className="platform-label" htmlFor="candidate-name">Candidate name</label>
                  <input
                    id="candidate-name"
                    className="platform-input"
                    value={selectedCandidate.name}
                    maxLength={80}
                    onChange={(event) => updateCandidateName(selectedCandidate.id, event.target.value)}
                    placeholder="Name this candidate"
                  />

                  <section className="template-box">
                    <div className="template-heading"><Database size={16} /><div><strong>Start from a dataset record</strong><span>Loads real observed feature values as an editable template.</span></div></div>
                    <div className="template-controls">
                      <label className="platform-label" htmlFor="template-index">Dataset record index (starts at 0)</label>
                      <div className="template-row">
                        <input
                          id="template-index"
                          className="platform-input"
                          type="number"
                          min="0"
                          step="1"
                          inputMode="numeric"
                          value={templateIndex}
                          onChange={(event) => setTemplateIndex(event.target.value)}
                        />
                        <button className="button secondary" onClick={() => void loadTemplate()} disabled={loadingTemplate || loadingFeatures}>
                          {loadingTemplate ? <><span className="spinner" /> Loading…</> : <>Load template <ArrowDown size={15} /></>}
                        </button>
                      </div>
                      {selectedCandidate.sourceIndex !== null && (
                        <div className="template-provenance">
                          <Check size={14} /> Modified candidate based on dataset row #{selectedCandidate.sourceIndex + 1}. The row’s recorded target is excluded.
                        </div>
                      )}
                    </div>
                  </section>

                  <div className="feature-progress-row">
                    <div><span>VALID FEATURE VALUES</span><strong>{loadingFeatures ? '—' : `${selectedCompleteness.supplied} / ${featureNames.length}`}</strong></div>
                    <div className="feature-progress-track"><i style={{ width: `${featureNames.length ? selectedCompleteness.supplied / featureNames.length * 100 : 0}%` }} /></div>
                  </div>
                  <div className="advanced-heading">
                    <div><span className="eyebrow tiny">ADVANCED INPUTS</span><h3>Model feature descriptors</h3></div>
                    <button className="button secondary small" aria-expanded={advancedOpen} onClick={() => setAdvancedOpen((open) => !open)}>
                      {advancedOpen ? 'Hide features' : 'Edit all 81 features'} <ChevronDown size={15} />
                    </button>
                  </div>
                  {!advancedOpen && (
                    <p className="advanced-hint">Enter values manually or load a dataset template. No feature values are prefilled unless they come from the selected real record.</p>
                  )}
                  {advancedOpen && (
                    <>
                      <div className="feature-search"><Search size={15} /><input aria-label="Search model feature names" placeholder="Filter descriptors…" value={globalSearch} onChange={(event) => setGlobalSearch(event.target.value)} /></div>
                      {groupsFiltered.map((group) => {
                        const valid = group.features.filter((feature) => {
                          const value = selectedCandidate.values[feature];
                          return value !== undefined && value !== '' && Number.isFinite(Number(value));
                        }).length;
                        return (
                          <details className="feature-group panel" key={group.name}>
                            <summary><span>{displayGroupName(group.name)}</span><span>{valid}/{group.features.length} filled</span><ChevronDown size={15} /></summary>
                            <div className="feature-input-grid">
                              {group.features.map((feature) => {
                                const value = selectedCandidate.values[feature] ?? '';
                                const range = featureRangeMap.get(feature);
                                const numericValue = value === '' ? NaN : Number(value);
                                const out = Number.isFinite(numericValue) &&
                                  ((isFiniteNumber(range?.min) && numericValue < range.min) ||
                                    (isFiniteNumber(range?.max) && numericValue > range.max));
                                return (
                                  <label className={`feature-input${out ? ' out-of-range' : ''}`} key={feature}>
                                    <span title={feature}>{displayName(feature)}</span>
                                    <input
                                      type="number"
                                      step="any"
                                      value={value}
                                      onChange={(event) => updateCandidateValue(selectedCandidate.id, feature, event.target.value)}
                                      aria-label={feature}
                                      aria-invalid={value !== '' && !Number.isFinite(Number(value))}
                                    />
                                    {out && <small>Outside observed dataset range</small>}
                                  </label>
                                );
                              })}
                            </div>
                          </details>
                        );
                      })}
                    </>
                  )}

                  {selectedCandidate.error && <div className="alert" role="alert">{selectedCandidate.error}</div>}
                  {apiError && featureNames.length === 81 && <div className="alert" role="alert">{apiError}</div>}
                  {apiError && featureNames.length !== 81 && <div className="alert api-retry" role="alert">{apiError}<button className="button secondary small" onClick={() => setConfigRetry((current) => current + 1)} disabled={loadingFeatures}>Retry connection</button></div>}
                  {explanationError && <div className="alert" role="alert">{explanationError}</div>}
                  {selectedOutOfRange.length > 0 && (
                    <div className="range-warning" role="status">
                      <CircleHelp size={17} />
                      <span><strong>{selectedOutOfRange.length} value{selectedOutOfRange.length === 1 ? '' : 's'} outside observed dataset ranges.</strong> This may affect reliability; in-range values do not guarantee accurate predictions.</span>
                    </div>
                  )}
                  {observedRangeCount < featureNames.length && summary && (
                    <div className="range-warning muted-warning"><CircleHelp size={16} /><span>Range checks are unavailable for {featureNames.length - observedRangeCount} model feature(s).</span></div>
                  )}

                  {selectedCandidate.prediction !== null && (
                    <section className="candidate-prediction panel">
                      <div className="candidate-prediction-label"><span className="eyebrow tiny">PREDICTED Tc · MODEL ESTIMATE</span><Thermometer size={17} /></div>
                      <strong>{formatNumber(selectedCandidate.prediction)} <small>K</small></strong>
                      {selectedCandidate.sourceIndex !== null && <span>Input template: dataset row #{selectedCandidate.sourceIndex + 1} · edited candidate values</span>}
                      <p>{PREDICTION_DISCLAIMER}</p>
                      <button className="button secondary small" onClick={() => { setExplainCandidateId(selectedCandidate.id); goTo('insights'); }}>Explain this prediction <ArrowRight size={14} /></button>
                    </section>
                  )}

                  <div className="discovery-actions">
                    <span><ShieldCheck size={15} /> Exactly 81 ordered numeric inputs. Recorded Tc is never sent.</span>
                    <button className="button primary" onClick={() => void predictCandidate(selectedCandidate)} disabled={loadingFeatures || featureNames.length !== 81 || !selectedCompleteness.complete || runningIds.includes(selectedCandidate.id)}>
                      {runningIds.includes(selectedCandidate.id) ? <><span className="spinner" /> Predicting…</> : <>Predict candidate Tc <ArrowRight size={16} /></>}
                    </button>
                  </div>
                  {selectedCompleteness.missing.length > 0 && featureNames.length === 81 && (
                    <p className="missing-hint">Missing {selectedCompleteness.missing.length} feature value{selectedCompleteness.missing.length === 1 ? '' : 's'}{advancedOpen ? `: ${selectedCompleteness.missing.slice(0, 4).map(displayName).join(', ')}${selectedCompleteness.missing.length > 4 ? ` and ${selectedCompleteness.missing.length - 4} more` : ''}` : '. Load a dataset template or enter all required values in Advanced Inputs.'}</p>
                  )}
                </section>

                <aside className="discovery-aside">
                  <section className="panel workflow-card">
                    <span className="workflow-icon"><Atom size={19} /></span>
                    <span className="eyebrow tiny">MODEL INPUT WORKFLOW</span>
                    <h3>Build a candidate, then estimate.</h3>
                    <p>A dataset row can supply a starting template. Edit its descriptors or supply a full 81-feature set manually. The source record’s measured Tc is not an input.</p>
                    <div className="workflow-model"><Activity size={15} /> Existing XGBoost regression model</div>
                  </section>
                  <section className="panel compare-prompt">
                    <div><span className="eyebrow tiny">MULTI-CANDIDATE WORKSPACE</span><strong>{candidates.length} profile{candidates.length === 1 ? '' : 's'}</strong></div>
                    <p>Compare predicted Tc only after candidates have valid model estimates.</p>
                    <button className="button secondary full" onClick={() => goTo('comparison')}>Open comparison <ArrowRight size={15} /></button>
                  </section>
                  <ReliabilityNote observedRangeCount={observedRangeCount} featureCount={featureNames.length} />
                </aside>
              </div>
            ) : <LoadingBox text={loadingFeatures ? 'Loading model feature schema…' : 'Create a candidate to begin.'} />}
          </>
        )}

        {page === 'comparison' && (
          <>
            <PageHead
              kicker="CANDIDATE WORKSPACE"
              title={<>Candidate <span>Comparison</span></>}
              sub="Compare model estimates and feature completeness across candidate profiles. Ranking is computational, not experimental evidence."
            />
            <section className="panel comparison-panel">
              <div className="comparison-toolbar">
                <div><span className="eyebrow tiny">PREDICTION RANKING</span><h2>Compare candidate estimates</h2></div>
                <div className="comparison-actions">
                  <button className="button secondary" onClick={addCandidate}><Plus size={14} /> New candidate</button>
                  <button className="button primary" onClick={() => void predictAllCompleteCandidates()} disabled={runningIds.length > 0 || !candidates.some((candidate) => candidateCompleteness(candidate.values, featureNames).complete)}>
                    {runningIds.length ? 'Running predictions…' : 'Predict all complete candidates'} <ArrowRight size={15} />
                  </button>
                </div>
              </div>
              {apiError && <div className="alert" role="alert">{apiError}</div>}
              {explanationError && <div className="alert" role="alert">{explanationError}</div>}
              {rankedCandidates.length > 0 ? (
                <div className="comparison-table-wrap">
                  <table className="comparison-table">
                    <thead><tr><th>Rank</th><th>Candidate</th><th>Predicted Tc</th><th>Difference to leader</th><th>Feature completeness</th><th>Actions</th></tr></thead>
                    <tbody>
                      {rankedCandidates.map((candidate, index) => {
                        const completeness = candidateCompleteness(candidate.values, featureNames);
                        const difference = (candidate.prediction ?? 0) - (rankedCandidates[0].prediction ?? 0);
                        return (
                          <tr key={candidate.id}>
                            <td><span className={`rank-badge${index === 0 ? ' first' : ''}`}>{index + 1}</span></td>
                            <td><strong>{candidate.name || 'Unnamed candidate'}</strong>{candidate.sourceIndex !== null && <small>Template row #{candidate.sourceIndex + 1}</small>}</td>
                            <td className="prediction-cell">{formatNumber(candidate.prediction ?? 0)} K</td>
                            <td>{index === 0 ? '—' : `${difference > 0 ? '+' : ''}${formatNumber(difference)} K`}</td>
                            <td><span className={completeness.complete ? 'complete-status' : 'incomplete-status'}>{completeness.supplied}/81 valid</span></td>
                            <td><div className="comparison-row-actions">
                              <button className="text-action" onClick={() => { selectCandidate(candidate.id); goTo('discovery'); }}>Edit</button>
                              <button className="text-action" onClick={() => void explain(candidate)} disabled={explainingId === candidate.id}>{explainingId === candidate.id ? 'Explaining…' : 'Explain'}</button>
                            </div></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  {rankedCandidates.map((candidate) => candidate.explanation && (
                    <LocalExplanation key={candidate.id} candidate={candidate} />
                  ))}
                </div>
              ) : (
                <EmptyState icon={<BarChart3 />} title="No candidate predictions yet" text="Create a candidate profile, supply all 81 valid features, and run a prediction to compare estimates." action={<button className="button primary" onClick={() => goTo('discovery')}>Build a candidate <ArrowRight size={15} /></button>} />
              )}
              {candidates.some((candidate) => candidate.prediction === null) && (
                <div className="candidate-completeness-list">
                  <span className="eyebrow tiny">OTHER CANDIDATE PROFILES</span>
                  {candidates.filter((candidate) => candidate.prediction === null).map((candidate) => {
                    const completeness = candidateCompleteness(candidate.values, featureNames);
                    return (
                      <div key={candidate.id}>
                        <span>{candidate.name || 'Unnamed candidate'}</span>
                        <span>{completeness.supplied}/81 valid features</span>
                        <button className="text-action" onClick={() => { selectCandidate(candidate.id); goTo('discovery'); }}>Complete profile</button>
                      </div>
                    );
                  })}
                </div>
              )}
              <div className="scientific-disclaimer"><ShieldCheck size={16} /><p>{DISCLAIMER}</p></div>
            </section>
          </>
        )}

        {page === 'dataset' && (
          <>
            <PageHead
              kicker="KNOWN SOURCE DATA"
              title={<>Dataset <span>Explorer</span></>}
              sub="Inspect real source rows and their recorded experimental target values. Recorded Tc is distinct from a candidate model prediction."
            />
            {summary && <div className="dataset-stats">
              <Stat label="SOURCE ROWS" value={summary.rows.toLocaleString()} icon={<Database />} />
              <Stat label="MODEL INPUT FEATURES" value={featureNames.length} icon={<Atom />} />
              <Stat label="MEAN RECORDED Tc" value={`${formatNumber(summary.target_stats.mean)} K`} icon={<Thermometer />} />
            </div>}
            <section className="panel dataset-panel">
              <div className="dataset-toolbar">
                <div><span className="eyebrow tiny">SOURCE CSV</span><h2>Recorded material descriptors</h2></div>
                <form className="dataset-search" onSubmit={(event) => { event.preventDefault(); void loadRecords(1, recordsQuery); }}>
                  <label htmlFor="target-search">Search recorded Tc</label>
                  <div><input id="target-search" type="number" step="any" value={recordsQuery} onChange={(event) => setRecordsQuery(event.target.value)} placeholder="Target in K" /><button className="button secondary small" type="submit" disabled={recordsLoading}>Search</button></div>
                </form>
              </div>
              {apiError && <div className="alert" role="alert">{apiError}</div>}
              <div className="table-scroll">
                <table className="dataset-table">
                  <thead><tr><th>Source row</th>{featureNames.slice(0, 6).map((feature) => <th key={feature}>{displayName(feature)}</th>)}<th>Recorded Tc</th><th>Template</th></tr></thead>
                  <tbody>{records.map((record) => (
                    <tr key={record.index}>
                      <td>#{record.index + 1}</td>
                      {featureNames.slice(0, 6).map((feature) => <td key={feature}>{record.values ? formatNumber(record.values[feature]) : 'Missing'}</td>)}
                      <td className="recorded-target">{record.target === null ? 'Missing' : `${formatNumber(record.target)} K`}</td>
                      <td><button className="text-action" disabled={!record.values} onClick={() => { setTemplateIndex(String(record.index)); goTo('discovery'); void loadTemplate(record.index, selectedCandidate?.id); }}>Use as candidate template</button></td>
                    </tr>
                  ))}</tbody>
                </table>
                {recordsLoading && <LoadingBox text="Loading source rows…" />}
                {!recordsLoading && records.length === 0 && <EmptyState icon={<Database />} title="No matching records" text="Try a different recorded Tc value or clear the search field." />}
              </div>
              <div className="dataset-pagination">
                <span>{recordsTotal.toLocaleString()} matching rows · page {recordsPage} of {Math.max(1, Math.ceil(recordsTotal / PAGE_SIZE))}</span>
                <div><button className="button secondary small" disabled={recordsLoading || recordsPage <= 1} onClick={() => void loadRecords(recordsPage - 1)}>Previous</button><button className="button secondary small" disabled={recordsLoading || recordsPage >= Math.ceil(recordsTotal / PAGE_SIZE)} onClick={() => void loadRecords(recordsPage + 1)}>Next</button></div>
              </div>
            </section>
          </>
        )}

        {page === 'insights' && (
          <>
            <PageHead
              kicker="MODEL TRANSPARENCY"
              title={<>Model <span>Insights</span></>}
              sub="Explore model-reported global feature importance, inspect local TreeSHAP contributions, and review feature values against the observed source-dataset ranges."
            />
            <div className="insights-grid">
              <section className="panel insights-card">
                <div className="insights-card-head"><Microscope size={18} /><div><span className="eyebrow tiny">GLOBAL MODEL INFORMATION</span><h2>{insights?.model ?? 'XGBoost regressor'}</h2></div></div>
                <p>Global importance summarizes how the fitted model uses each feature across its trees. It is not a per-candidate explanation and does not establish causality.</p>
                {insights ? (
                  <div className="importance-list">
                    {insights.importance.slice(0, 12).map((entry) => {
                      const maximum = Math.max(...insights.importance.map((item) => item.importance), 1e-12);
                      return <div className="importance-row" key={entry.feature}><span>{displayName(entry.feature)}</span><div><i style={{ width: `${entry.importance / maximum * 100}%` }} /></div><strong>{formatNumber(entry.importance)}</strong></div>;
                    })}
                  </div>
                ) : <LoadingBox text="Global model importance is unavailable." />}
                <span className="importance-note">Model-reported global feature importance · not a causal measure</span>
              </section>

              <section className="panel insights-card local-shap-card">
                <div className="insights-card-head"><Activity size={18} /><div><span className="eyebrow tiny">PER-CANDIDATE EXPLANATION</span><h2>TreeSHAP contributions</h2></div></div>
                <p>TreeSHAP values decompose a single prediction into feature contributions relative to the model baseline. Contributions explain model output, not scientific causation.</p>
                <label className="platform-label" htmlFor="explain-candidate">Predicted candidate</label>
                <select id="explain-candidate" className="platform-input" value={explainCandidate?.id ?? ''} onChange={(event) => setExplainCandidateId(event.target.value)}>
                  <option value="">Select a candidate</option>
                  {candidates.filter((candidate) => candidateCompleteness(candidate.values, featureNames).complete).map((candidate) => <option key={candidate.id} value={candidate.id}>{candidate.name}{candidate.prediction === null ? ' · not predicted' : ` · ${formatNumber(candidate.prediction)} K`}</option>)}
                </select>
                <button className="button primary explain-button" onClick={() => explainCandidate && void explain(explainCandidate)} disabled={!explainCandidate || !candidateCompleteness(explainCandidate.values, featureNames).complete || explainingId === explainCandidate?.id}>
                  {explainingId === explainCandidate?.id ? <><span className="spinner" /> Computing…</> : 'Compute candidate explanation'}
                </button>
                {explanationError && <div className="alert" role="alert">{explanationError}</div>}
                {explainCandidate?.explanation && <LocalExplanation candidate={explainCandidate} />}
                <p className="method-note">If the model cannot provide contributions that reconcile with its prediction, the explanation is reported unavailable.</p>
              </section>
            </div>
            <section className="panel reliability-card">
              <div className="insights-card-head"><CircleHelp size={18} /><div><span className="eyebrow tiny">PREDICTION RELIABILITY CHECK</span><h2>Observed dataset ranges</h2></div></div>
              <p>Input values outside the source CSV’s observed minimum–maximum ranges are flagged. These bounds describe this dataset; being within range does not guarantee a reliable prediction or indicate that a candidate is chemically feasible.</p>
              <div className="reliability-summary"><strong>{observedRangeCount} / {featureNames.length || 81}</strong><span>features have observed min/max statistics</span><strong>{selectedCandidate ? selectedOutOfRange.length : '—'}</strong><span>selected candidate values outside those bounds</span></div>
              {selectedCandidate && selectedOutOfRange.length > 0 && <div className="range-feature-list">{selectedOutOfRange.slice(0, 12).map((item) => <span key={item.feature}>{displayName(item.feature)}: {formatNumber(item.value)} ({item.direction} {item.direction === 'below' ? formatNumber(item.min ?? item.value) : formatNumber(item.max ?? item.value)})</span>)}</div>}
              {featureRanges.length === 0 && <LoadingBox text="Source dataset range statistics are currently unavailable." />}
            </section>
            <section className="panel evaluation-card">
              <div className="insights-card-head"><Check size={18} /><div><span className="eyebrow tiny">ORIGINAL NOTEBOOK EVALUATION</span><h2>Reported holdout results</h2></div></div>
              <div className="evaluation-metrics">
                <div><span>R² SCORE</span><strong>0.9264</strong></div>
                <div><span>MAE</span><strong>5.5566 K</strong></div>
                <div><span>RMSE</span><strong>9.3099 K</strong></div>
              </div>
              <p>Reported in the original notebook for its 80/20 split (random_state 42). These are not a live evaluation of the deployed artifact and do not guarantee performance on unseen candidates.</p>
            </section>
          </>
        )}

        {page === 'history' && (
          <>
            <PageHead kicker="LOCAL WORKSPACE" title={<>Prediction <span>History</span></>} sub="Candidate predictions are saved in this browser only. Historical entries are model estimates, not experimental labels." />
            <section className="panel history-panel">
              {history.length ? history.map((item) => (
                <div className="history-row" key={item.id}>
                  <Atom size={18} />
                  <div><strong>{item.candidateName}</strong><span>{formatNumber(item.prediction)} K · Predicted Tc · {new Date(item.at).toLocaleString()}{item.sourceIndex !== null ? ` · template row #${item.sourceIndex + 1}` : ''}</span></div>
                  <button className="text-action" onClick={() => {
                    const existing = candidates.find((candidate) => candidate.name === item.candidateName);
                    if (existing) {
                      selectCandidate(existing.id);
                      goTo('discovery');
                    } else if (item.values) {
                      const candidate = {
                        ...newCandidate(candidates.length + 1),
                        name: item.candidateName,
                        sourceIndex: item.sourceIndex,
                        values: item.values,
                        prediction: item.prediction,
                      };
                      setCandidates((current) => [...current, candidate]);
                      setSelectedCandidateId(candidate.id);
                      goTo('discovery');
                    }
                  }}>{item.values ? 'Restore candidate' : 'Saved estimate'}</button>
                </div>
              )) : <EmptyState icon={<History />} title="No predictions saved" text="Predictions from Material Discovery will appear here." action={<button className="button primary" onClick={() => goTo('discovery')}>Build a candidate <ArrowRight size={15} /></button>} />}
              {history.length > 0 && <button className="button secondary small clear-history" onClick={() => setHistory([])}><RotateCcw size={14} /> Clear history</button>}
            </section>
          </>
        )}
      </main>

      <footer className="platform-footer page-width">
        <span>CRYONEX · QUANTUM MATERIALS INTELLIGENCE</span>
        <span>MODEL ESTIMATES ARE NOT EXPERIMENTAL VERIFICATION</span>
      </footer>
    </div>
  );
}

function PageHead({ kicker, title, sub }: { kicker: string; title: ReactNode; sub: string }) {
  return <div className="platform-page-head"><span className="eyebrow"><i />{kicker}</span><h1>{title}</h1><p>{sub}</p></div>;
}

function Stat({ label, value, icon }: { label: string; value: string | number; icon: ReactNode }) {
  return <div className="panel stat-card"><span className="stat-icon">{icon}</span><span>{label}</span><strong>{value}</strong></div>;
}

function LoadingBox({ text }: { text: string }) {
  return <div className="panel loading-box"><span className="spinner" />{text}</div>;
}

function EmptyState({ icon, title, text, action }: { icon: ReactNode; title: string; text: string; action?: ReactNode }) {
  return <div className="empty-state"><span>{icon}</span><h3>{title}</h3><p>{text}</p>{action}</div>;
}

function ReliabilityNote({ observedRangeCount, featureCount }: { observedRangeCount: number; featureCount: number }) {
  return <div className="range-note"><CircleHelp size={16} /><p>Observed dataset ranges flag unusual inputs; they are not confidence bounds. In-range values do not guarantee a reliable estimate.</p><span>{observedRangeCount}/{featureCount || 81} ranges available</span></div>;
}

function LocalExplanation({ candidate }: { candidate: CandidateProfile }) {
  const explanation = candidate.explanation;
  if (!explanation) return null;
  const strongest = [...explanation.contributions]
    .sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution))
    .slice(0, 10);
  const maxMagnitude = Math.max(...strongest.map((entry) => Math.abs(entry.contribution)), 1e-12);
  return (
    <section className="local-explanation">
      <div className="explanation-heading"><span className="eyebrow tiny">LOCAL EXPLANATION · {candidate.name}</span><span>Base value {formatNumber(explanation.base_value)} K</span></div>
      <p>{explanation.method}. Showing the 10 largest absolute contributions for this candidate; positive values raise this model output relative to baseline, negative values lower it.</p>
      <div className="contribution-list">
        {strongest.map((entry) => (
          <div className="contribution-row" key={entry.feature}>
            <span>{entry.feature.replace(/_/g, ' ')}</span>
            <div className="contribution-track"><i className={entry.contribution >= 0 ? 'positive' : 'negative'} style={{ width: `${Math.abs(entry.contribution) / maxMagnitude * 100}%` }} /></div>
            <strong className={entry.contribution >= 0 ? 'positive-text' : 'negative-text'}>{entry.contribution > 0 ? '+' : ''}{formatNumber(entry.contribution)} K</strong>
          </div>
        ))}
      </div>
      <div className="explanation-result">Baseline + contributions = model prediction <strong>{formatNumber(explanation.prediction)} K</strong></div>
    </section>
  );
}
