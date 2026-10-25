import { useEffect, useMemo, useState } from 'react';
import { Activity, Atom, ArrowLeft, ArrowRight, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Database, FlaskConical, History, LayoutDashboard, Microscope, RotateCcw, ShieldCheck, Thermometer } from 'lucide-react';
import './quick.css';

const API = (import.meta.env.VITE_API_BASE_URL || 'https://cyronex.onrender.com').replace(/\/$/, '');
const PAGE_SIZE = 20;
const REQUEST_TIMEOUT_MS = 60_000;
const DISCLAIMER = 'This is a machine-learning prediction based on the supplied material features. It is an estimate and does not constitute experimental confirmation.';

type Page = 'quick' | 'result' | 'overview' | 'explorer' | 'insights' | 'history';
type QuickRecord = {
  index: number;
  features: Record<string, number> | null;
  criticalTemp: number | null;
};
type HistoryItem = {
  id: string;
  at: string;
  prediction: number;
  recordIndex?: number;
  features?: Record<string, number>;
  criticalTemp?: number | null;
};
type DatasetSummary = {
  rows: number;
  features: number;
  missing_values: number;
  target_stats: { min: number; max: number; mean: number; median: number };
};
type ModelInsights = {
  model: string;
  feature_count: number;
  importance: Array<{ feature: string; importance: number }>;
};

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
        typeof payload === 'object' &&
        payload !== null &&
        'error' in payload &&
        typeof payload.error === 'string'
          ? payload.error
          : `The API returned an error (${response.status}).`;
      throw new Error(message);
    }
    if (payload === null) throw new Error('The API returned an unreadable response.');
    return payload as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('The API took too long to respond. Please try again.');
    }
    if (error instanceof TypeError) {
      throw new Error('Could not reach the CryoNex API. Check your connection and try again.');
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function readFeatures(response: unknown): string[] {
  if (
    typeof response !== 'object' ||
    response === null ||
    !('features' in response) ||
    !Array.isArray(response.features) ||
    response.features.length !== 81 ||
    new Set(response.features).size !== 81 ||
    response.features.some((feature) => typeof feature !== 'string')
  ) {
    throw new Error('The API did not return the model’s complete set of 81 features.');
  }
  return response.features;
}

function toRecords(
  response: unknown,
  features: string[],
): { records: QuickRecord[]; page: number; total: number } {
  if (
    typeof response !== 'object' ||
    response === null ||
    !('rows' in response) ||
    !Array.isArray(response.rows) ||
    !('page' in response) ||
    typeof response.page !== 'number' ||
    !Number.isInteger(response.page) ||
    !('page_size' in response) ||
    typeof response.page_size !== 'number' ||
    !Number.isInteger(response.page_size) ||
    response.page_size < 1 ||
    !('total' in response) ||
    typeof response.total !== 'number' ||
    !Number.isInteger(response.total) ||
    response.rows.some(
      (row) => typeof row !== 'object' || row === null || Array.isArray(row),
    )
  ) {
    throw new Error('The dataset API returned an unexpected response.');
  }
  const data = response as {
    rows: Array<Record<string, unknown>>;
    page: number;
    page_size: number;
    total: number;
  };
  return {
    page: data.page,
    total: data.total,
    records: data.rows.map((row, index) => {
      const values = features.map((feature) => row[feature]);
      const complete = values.every(
        (value): value is number => typeof value === 'number' && Number.isFinite(value),
      );
      const target = row.critical_temp;
      return {
        index: (data.page - 1) * data.page_size + index,
        features: complete
          ? Object.fromEntries(features.map((feature, i) => [feature, values[i]]))
          : null,
        criticalTemp:
          typeof target === 'number' && Number.isFinite(target) ? target : null,
      };
    }),
  };
}

function toSampleRecord(response: unknown, featureNames: string[]): QuickRecord {
  if (
    !isRecord(response) ||
    !('index' in response) ||
    typeof response.index !== 'number' ||
    !Number.isInteger(response.index) ||
    !('features' in response) ||
    !isRecord(response.features) ||
    Object.keys(response.features).length !== featureNames.length ||
    !('critical_temp' in response) ||
    !isFiniteNumber(response.critical_temp)
  ) {
    throw new Error('This dataset sample is incomplete and cannot be used for prediction.');
  }
  const orderedFeatures: Record<string, number> = {};
  for (const feature of featureNames) {
    const value = response.features[feature];
    if (!isFiniteNumber(value)) {
      throw new Error('This dataset sample is incomplete and cannot be used for prediction.');
    }
    orderedFeatures[feature] = value;
  }
  return {
    index: response.index,
    features: orderedFeatures,
    criticalTemp: response.critical_temp,
  };
}

function readPrediction(response: unknown): number {
  if (
    typeof response !== 'object' ||
    response === null ||
    !('predicted_critical_temperature_K' in response) ||
    typeof response.predicted_critical_temperature_K !== 'number' ||
    !Number.isFinite(response.predicted_critical_temperature_K)
  ) {
    throw new Error('The API did not return a valid predicted temperature.');
  }
  return response.predicted_critical_temperature_K;
}

function readHistory(): HistoryItem[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem('cryonex-history') || '[]');
    if (!Array.isArray(stored)) return [];
    return stored.filter(
      (item): item is HistoryItem =>
        typeof item === 'object' &&
        item !== null &&
        typeof item.id === 'string' &&
        typeof item.at === 'string' &&
        typeof item.prediction === 'number' &&
        Number.isFinite(item.prediction) &&
        (item.recordIndex === undefined ||
          (Number.isInteger(item.recordIndex) && item.recordIndex >= 0)) &&
        (item.features === undefined ||
          (typeof item.features === 'object' &&
            item.features !== null &&
            !Array.isArray(item.features) &&
            Object.keys(item.features).length === 81 &&
            Object.values(item.features).every(
              (value) => typeof value === 'number' && Number.isFinite(value),
            ))),
    );
  } catch {
    return [];
  }
}

const fmt = (value: number) =>
  new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value);
const nice = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

const navigation = [
  { id: 'quick' as const, label: 'Quick prediction', icon: FlaskConical },
  { id: 'overview' as const, label: 'Overview', icon: LayoutDashboard },
  { id: 'explorer' as const, label: 'Dataset', icon: Database },
  { id: 'insights' as const, label: 'Model insights', icon: Microscope },
  { id: 'history' as const, label: 'History', icon: History },
];

export default function App() {
  const [page, setPage] = useState<Page>('quick');
  const [retryToken, setRetryToken] = useState(0);
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [records, setRecords] = useState<QuickRecord[]>([]);
  const [recordPage, setRecordPage] = useState(1);
  const [recordTotal, setRecordTotal] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [loadingRecords, setLoadingRecords] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState<number | null>(null);
  const [resultRecord, setResultRecord] = useState<QuickRecord | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [insights, setInsights] = useState<ModelInsights | null>(null);
  const [showFeatures, setShowFeatures] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>(readHistory);

  const selectedRecord = records.find((record) => record.index === selectedIndex) ?? null;
  const pageCount = Math.max(1, Math.ceil(recordTotal / PAGE_SIZE));
  const topFeatures = useMemo(
    () => insights?.importance.slice(0, 5) ?? [],
    [insights],
  );

  async function loadRecords(nextPage: number, names = featureNames) {
    setLoadingRecords(true);
    setError('');
    try {
      const payload = await requestJSON<unknown>(
        `/dataset/records?page=${nextPage}&page_size=${PAGE_SIZE}&q=`,
      );
      const result = toRecords(payload, names);
      setRecords(result.records);
      setRecordPage(result.page);
      setRecordTotal(result.total);
      setSelectedIndex((current) =>
        result.records.some((record) => record.index === current && record.features)
          ? current
          : result.records.find((record) => record.features !== null)?.index ?? null,
      );
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : 'Dataset records could not be loaded. Please try again.',
      );
    } finally {
      setLoadingRecords(false);
    }
  }

  useEffect(() => {
    let active = true;
    async function initialize() {
      try {
        const response = await requestJSON<unknown>('/features');
        const names = readFeatures(response);
        if (!active) return;
        setFeatureNames(names);
        await loadRecords(1, names);
      } catch (loadError) {
        if (!active) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : 'CryoNex could not load the model features.',
        );
        setLoadingRecords(false);
      }
    }
    void initialize();

    void requestJSON<DatasetSummary>('/dataset/summary')
      .then((data) => active && setSummary(data))
      .catch(() => undefined);
    void requestJSON<ModelInsights>('/model/insights')
      .then((data) => active && setInsights(data))
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [retryToken]);

  function selectRecord(index: number) {
    setSelectedIndex(index);
    setPrediction(null);
    setResultRecord(null);
    setError('');
  }

  async function loadSample() {
    setLoadingRecords(true);
    setError('');
    try {
      const index = selectedIndex ?? 0;
      const response = await requestJSON<unknown>(`/sample?index=${index}`);
      if (!featureNames.length) throw new Error('Model features are not available. Please retry.');
      const sampleRecord = toSampleRecord(response, featureNames);
      setRecords((current) => [
        sampleRecord,
        ...current.filter((record) => record.index !== sampleRecord.index),
      ]);
      setSelectedIndex(sampleRecord.index);
      setPrediction(null);
    } catch (sampleError) {
      setError(
        sampleError instanceof Error
          ? sampleError.message
          : 'A valid dataset sample could not be loaded.',
      );
    } finally {
      setLoadingRecords(false);
    }
  }

  async function predict() {
    setError('');
    if (!selectedRecord?.features || featureNames.length !== 81) {
      setError('Select a complete dataset record before requesting a prediction.');
      return;
    }
    const payload = Object.fromEntries(
      featureNames.map((feature) => [feature, selectedRecord.features?.[feature]]),
    );
    if (
      Object.keys(payload).length !== 81 ||
      featureNames.some(
        (feature) =>
          typeof payload[feature] !== 'number' || !Number.isFinite(payload[feature]),
      )
    ) {
      setError('This record does not contain all 81 valid model features.');
      return;
    }

    setPredicting(true);
    try {
      const response = await requestJSON<unknown>(
        '/predict',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        },
      );
      const predictedTemperature = readPrediction(response);
      setPrediction(predictedTemperature);
      setResultRecord({
        ...selectedRecord,
        features: { ...selectedRecord.features },
      });
      setShowFeatures(false);
      setPage('result');
      const item: HistoryItem = {
        id: crypto.randomUUID(),
        at: new Date().toISOString(),
        prediction: predictedTemperature,
        recordIndex: selectedRecord.index,
        features: { ...selectedRecord.features },
        criticalTemp: selectedRecord.criticalTemp,
      };
      const next = [item, ...history].slice(0, 100);
      setHistory(next);
      localStorage.setItem('cryonex-history', JSON.stringify(next));
    } catch (predictionError) {
      setError(
        predictionError instanceof Error
          ? predictionError.message
          : 'The prediction could not be completed. Please try again.',
      );
    } finally {
      setPredicting(false);
    }
  }

  function goTo(nextPage: Page) {
    setPage(nextPage);
    setError('');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  return (
    <div className="app-shell quick-app">
      <header className="topbar quick-topbar">
        <button className="brand quick-brand" onClick={() => goTo('quick')} aria-label="CryoNex quick prediction">
          <span className="brand-mark"><Atom size={19} /></span>
          <span>CRYO<span className="brand-accent">NEX</span><small>QUANTUM MATERIALS INTELLIGENCE</small></span>
        </button>
        <nav className="desktop-nav quick-nav" aria-label="Main navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={page === id ? 'active' : ''}
              onClick={() => goTo(id)}
              aria-current={page === id ? 'page' : undefined}
            >
              <Icon size={15} />{label}
            </button>
          ))}
        </nav>
        <div className="quick-status"><i /> Saved XGBoost model</div>
      </header>

      <main className="quick-main page-width">
        {page === 'quick' && (
          <>
            <section className="quick-intro">
              <span className="eyebrow"><span /> QUICK PREDICTION</span>
              <h1>Explore a material.<br /><span>Estimate its critical temperature.</span></h1>
              <p>Choose a real superconductivity dataset record. CryoNex will send its complete set of model features to the saved XGBoost estimator.</p>
            </section>

            <section className="quick-workspace">
              <div className="quick-selector panel">
                <div className="quick-section-heading">
                  <span className="quick-step">01</span>
                  <div><span className="eyebrow tiny">DATASET RECORD</span><h2>Select a material record</h2></div>
                  <Database size={19} />
                </div>

                {loadingRecords && records.length === 0 ? (
                  <LoadingBox text="Connecting to the dataset…" />
                ) : (
                  <>
                    <label className="quick-label" htmlFor="material-record">Available records</label>
                    <div className="quick-select-wrap">
                      <select
                        id="material-record"
                        value={selectedIndex ?? ''}
                        onChange={(event) => selectRecord(Number(event.target.value))}
                        disabled={loadingRecords || !records.some((record) => record.features)}
                      >
                        {!records.some((record) => record.features) && <option value="">No complete records on this page</option>}
                        {records.map((record) => (
                          <option
                            key={record.index}
                            value={record.index}
                            disabled={!record.features}
                          >
                            Dataset record #{record.index + 1}{record.criticalTemp === null ? '' : ` · recorded Tc ${fmt(record.criticalTemp)} K`}{!record.features ? ' · incomplete features' : ''}
                          </option>
                        ))}
                      </select>
                      <ChevronDown size={16} aria-hidden="true" />
                    </div>
                    <div className="quick-paging">
                      <span>{recordTotal.toLocaleString()} source records</span>
                      <div>
                        <button className="icon-button" aria-label="Previous records" onClick={() => void loadRecords(recordPage - 1)} disabled={loadingRecords || recordPage <= 1}><ChevronLeft size={16} /></button>
                        <span>Page {recordPage} of {pageCount}</span>
                        <button className="icon-button" aria-label="Next records" onClick={() => void loadRecords(recordPage + 1)} disabled={loadingRecords || recordPage >= pageCount}><ChevronRight size={16} /></button>
                      </div>
                    </div>
                  </>
                )}

                {selectedRecord?.features ? (
                  <div className="quick-record-summary">
                    <div className="quick-record-title">
                      <span className="quick-record-mark"><Atom size={19} /></span>
                      <div><span>SELECTED SOURCE ROW</span><strong>Record #{selectedRecord.index + 1}</strong></div>
                      {selectedRecord.criticalTemp !== null && <span className="quick-target">Dataset Tc {fmt(selectedRecord.criticalTemp)} K</span>}
                    </div>
                    <p>The source dataset provides engineered numerical descriptors, not a material name or chemical formula. This prediction uses the selected row’s actual feature values.</p>
                    <div className="quick-properties">
                      {['number_of_elements', 'mean_atomic_mass', 'mean_atomic_radius'].map((feature) => (
                        <div key={feature}><span>{nice(feature)}</span><strong>{selectedRecord.features?.[feature] === undefined ? '—' : fmt(selectedRecord.features[feature])}</strong></div>
                      ))}
                    </div>
                    <button className="quick-text-action" onClick={() => void loadSample()} disabled={loadingRecords}>
                      <RotateCcw size={14} /> {loadingRecords ? 'Loading sample…' : `Reload this record from /sample`}
                    </button>
                  </div>
                ) : (
                  <div className="quick-empty">
                    <CircleHelp size={18} />
                    <span>{records.length ? 'No complete 81-feature records are available on this page. Try another page.' : 'Dataset records are unavailable. Retry to connect to the backend.'}</span>
                    <button className="button secondary small" onClick={() => featureNames.length === 81 ? void loadRecords(recordPage) : setRetryToken((current) => current + 1)} disabled={loadingRecords}>Retry</button>
                  </div>
                )}

                {error && <div className="alert" role="alert">{error}</div>}
                <div className="quick-predict-row">
                  <div><ShieldCheck size={16} /><span>81 validated features · no generated inputs</span></div>
                  <button className="button primary quick-predict-button" onClick={() => void predict()} disabled={!selectedRecord?.features || predicting}>
                    {predicting ? <><span className="spinner" /> Predicting…</> : <>Predict Critical Temperature <ArrowRight size={16} /></>}
                  </button>
                </div>
              </div>

              <aside className="quick-side">
                <div className="quick-info-card panel">
                  <span className="quick-info-icon"><Activity size={19} /></span>
                  <span className="eyebrow tiny">HOW IT WORKS</span>
                  <h3>Real data in, model estimate out.</h3>
                  <p>The selected source row includes all 81 features expected by the trained model. Only those feature values are sent for prediction; the dataset target is not included in the input.</p>
                  <div className="quick-model-tag"><Atom size={15} /> XGBoost regression</div>
                </div>
                {summary && (
                  <div className="quick-dataset-card panel">
                    <span className="eyebrow tiny">SOURCE DATASET</span>
                    <strong>{summary.rows.toLocaleString()}</strong>
                    <span>records · {summary.features} model features</span>
                    <div>Target range: {fmt(summary.target_stats.min)}–{fmt(summary.target_stats.max)} K</div>
                  </div>
                )}
              </aside>
            </section>
          </>
        )}

        {page === 'result' && (
          <section className="quick-result-page">
            <button className="quick-back" onClick={() => goTo('quick')}><ArrowLeft size={15} /> Back to Quick Prediction</button>
            <div className="quick-result-card panel">
              <div className="quick-result-label"><span className="eyebrow"><span /> MODEL ESTIMATE</span><span className="unit-pill">KELVIN</span></div>
              <div className="quick-temperature">{prediction === null ? '—' : fmt(prediction)}<span>K</span></div>
              <h1>Predicted critical temperature</h1>
              <div className="quick-result-record">
                <span className="quick-record-mark"><Atom size={18} /></span>
                <div><span>INPUT DATASET RECORD</span><strong>{resultRecord ? `Record #${resultRecord.index + 1}` : 'Dataset record unavailable'}</strong></div>
                {resultRecord?.criticalTemp !== null && resultRecord?.criticalTemp !== undefined && (
                  <div className="quick-result-target"><span>RECORDED DATASET Tc</span><strong>{fmt(resultRecord.criticalTemp)} K</strong></div>
                )}
              </div>
              <p className="quick-result-explainer">This estimate was generated from the selected record’s complete set of {resultRecord?.features ? featureNames.length : 0} input features.</p>
              <details className="quick-feature-details" open={showFeatures} onToggle={(event) => setShowFeatures(event.currentTarget.open)}>
                <summary>View the 81 input features <ChevronDown size={16} /></summary>
                {resultRecord?.features && (
                  <div className="quick-feature-list">
                    {featureNames.map((feature) => (
                      <div key={feature}><span>{nice(feature)}</span><strong>{fmt(resultRecord.features![feature])}</strong></div>
                    ))}
                  </div>
                )}
              </details>
              <div className="quick-disclaimer"><ShieldCheck size={17} /><p>{DISCLAIMER}</p></div>
              <button className="button primary quick-another" onClick={() => goTo('quick')}>Predict Another Material <ArrowRight size={16} /></button>
            </div>
            {insights && (
              <div className="quick-result-model">
                <Microscope size={16} />
                <span>Model information</span>
                <strong>{insights.model} · {insights.feature_count} input features</strong>
              </div>
            )}
          </section>
        )}

        {page === 'overview' && (
          <section className="quick-secondary-page">
            <PageHead kicker="CRYONEX" title="Quantum materials intelligence" sub="Explore superconductivity dataset records and request model estimates from a saved XGBoost model." />
            <button className="button primary" onClick={() => goTo('quick')}>Start a Quick Prediction <ArrowRight size={16} /></button>
            <div className="quick-overview-stats">
              <Stat label="DATASET RECORDS" value={summary ? summary.rows.toLocaleString() : '—'} icon={<Database />} />
              <Stat label="MODEL FEATURES" value={featureNames.length || '—'} icon={<Atom />} />
              <Stat label="ESTIMATOR" value={insights?.model ?? '—'} icon={<Activity />} />
            </div>
          </section>
        )}

        {page === 'explorer' && (
          <section className="quick-secondary-page">
            <PageHead kicker="DATASET EXPLORER" title="Explore source records" sub="Browse real rows from the superconductivity dataset. Select any complete record to use it in Quick Prediction." />
            {error && <div className="alert" role="alert">{error}</div>}
            <div className="quick-dataset-card panel">
              <div className="quick-explorer-stats">
                <Stat label="SOURCE RECORDS" value={summary ? summary.rows.toLocaleString() : '—'} icon={<Database />} />
                <Stat label="MODEL FEATURES" value={featureNames.length || '—'} icon={<Atom />} />
                <Stat label="MEAN DATASET Tc" value={summary ? `${fmt(summary.target_stats.mean)} K` : '—'} icon={<Thermometer />} />
              </div>
              <div className="quick-table-wrap">
                <table>
                  <thead><tr><th>Record</th>{featureNames.slice(0, 5).map((feature) => <th key={feature}>{nice(feature)}</th>)}<th>Recorded Tc</th><th>Action</th></tr></thead>
                  <tbody>
                    {records.map((record) => (
                      <tr key={record.index}>
                        <td>#{record.index + 1}</td>
                        {featureNames.slice(0, 5).map((feature) => <td key={feature}>{record.features ? fmt(record.features[feature]) : '—'}</td>)}
                        <td>{record.criticalTemp === null ? '—' : `${fmt(record.criticalTemp)} K`}</td>
                        <td><button className="quick-table-action" disabled={!record.features} onClick={() => { selectRecord(record.index); goTo('quick'); }}>Select</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {loadingRecords && <LoadingBox text="Loading dataset records…" />}
              </div>
              <div className="quick-paging">
                <span>{recordTotal.toLocaleString()} source records</span>
                <div>
                  <button className="button secondary small" onClick={() => void loadRecords(recordPage - 1)} disabled={loadingRecords || recordPage <= 1}>Previous</button>
                  <span>Page {recordPage} of {pageCount}</span>
                  <button className="button secondary small" onClick={() => void loadRecords(recordPage + 1)} disabled={loadingRecords || recordPage >= pageCount}>Next</button>
                </div>
              </div>
            </div>
          </section>
        )}

        {page === 'insights' && (
          <section className="quick-secondary-page">
            <PageHead kicker="MODEL INSIGHTS" title="Inside the estimator" sub="Information reported by the saved model artifact and current API." />
            {insights ? (
              <div className="quick-insights panel">
                <div className="quick-info-icon"><Microscope size={20} /></div>
                <span className="eyebrow tiny">MODEL ARTIFACT</span>
                <h2>{insights.model}</h2>
                <p>{insights.feature_count} input features · predictions are estimates, not experimental measurements.</p>
                <h3>Most important model features</h3>
                {topFeatures.map((feature) => (
                  <div className="quick-importance" key={feature.feature}>
                    <span>{nice(feature.feature)}</span>
                    <div><i style={{ width: `${Math.max(0, Math.min(100, feature.importance * 100))}%` }} /></div>
                    <strong>{fmt(feature.importance)}</strong>
                  </div>
                ))}
              </div>
            ) : <LoadingBox text="Model insights are not available right now." />}
          </section>
        )}

        {page === 'history' && (
          <section className="quick-secondary-page">
            <PageHead kicker="LOCAL HISTORY" title="Recent predictions" sub="This list is saved in this browser only and is not sent to the backend." />
            <div className="quick-history panel">
              {history.length ? history.map((item) => (
                <div className="quick-history-row" key={item.id}>
                  <Atom size={17} />
                  <div><strong>{fmt(item.prediction)} K</strong><span>{item.recordIndex === undefined ? 'Historical estimate' : `Record #${item.recordIndex + 1}`} · {new Date(item.at).toLocaleString()}</span></div>
                  <button className="icon-button" aria-label="View saved prediction" onClick={() => {
                    const restored = item.features
                      ? { index: item.recordIndex ?? 0, features: item.features, criticalTemp: item.criticalTemp ?? null }
                      : records.find((record) => record.index === item.recordIndex) ?? null;
                    setResultRecord(restored);
                    setPrediction(item.prediction);
                    goTo('result');
                  }}><ArrowRight size={16} /></button>
                </div>
              )) : <div className="quick-empty"><History size={19} /> No predictions yet. Choose a dataset record to get started.</div>}
              {history.length > 0 && <button className="button secondary small" onClick={() => { setHistory([]); localStorage.removeItem('cryonex-history'); }}>Clear history</button>}
            </div>
          </section>
        )}
      </main>

      <footer className="global-footer page-width"><span>CRYONEX · QUANTUM MATERIALS INTELLIGENCE</span><span>MODEL ESTIMATES ARE NOT EXPERIMENTAL VERIFICATION</span></footer>
    </div>
  );
}

function PageHead({ kicker, title, sub }: { kicker: string; title: string; sub: string }) {
  return <div className="page-head"><span className="eyebrow"><span />{kicker}</span><h1>{title}</h1><p>{sub}</p></div>;
}

function Stat({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
  return <div className="panel stat-card"><div className="stat-icon">{icon}</div><span>{label}</span><strong>{value}</strong></div>;
}

function LoadingBox({ text }: { text: string }) {
  return <div className="panel loading-box"><span className="spinner" />{text}</div>;
}
