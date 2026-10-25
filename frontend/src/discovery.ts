export type CandidateProfile = {
  id: string;
  name: string;
  sourceIndex: number | null;
  values: Record<string, number | string>;
  prediction: number | null;
  explanation?: CandidateExplanation;
  error?: string;
};

export type CandidateExplanation = {
  method: string;
  prediction: number;
  base_value: number;
  contributions: Array<{
    feature: string;
    value: number;
    contribution: number;
  }>;
};

export type FeatureRange = {
  feature: string;
  min: number | null;
  max: number | null;
};

export function candidateCompleteness(
  values: Record<string, unknown>,
  featureNames: string[],
) {
  const missing: string[] = [];
  const invalid: string[] = [];
  for (const feature of featureNames) {
    const value = values[feature];
    if (value === undefined || value === '') {
      missing.push(feature);
    } else if (typeof value === 'string' && value.trim() === '') {
      missing.push(feature);
    } else if (
      (typeof value !== 'number' && typeof value !== 'string') ||
      !Number.isFinite(Number(value))
    ) {
      invalid.push(feature);
    }
  }
  return {
    complete: featureNames.length === 81 && missing.length === 0 && invalid.length === 0,
    supplied: featureNames.length - missing.length - invalid.length,
    missing,
    invalid,
  };
}

export function orderedFeaturePayload(
  values: Record<string, unknown>,
  featureNames: string[],
): Record<string, number> {
  const completeness = candidateCompleteness(values, featureNames);
  if (!completeness.complete) {
    const unresolved = [...completeness.missing, ...completeness.invalid];
    throw new Error(
      unresolved.length
        ? `Complete or correct these features before predicting: ${unresolved.join(', ')}.`
        : 'The model feature list must contain exactly 81 features.',
    );
  }
  return Object.fromEntries(
    featureNames.map((feature) => [feature, Number(values[feature])]),
  );
}

export function rankCandidates<T extends { prediction: number | null; name: string }>(
  candidates: T[],
): T[] {
  return candidates
    .filter(
      (candidate): candidate is T & { prediction: number } =>
        candidate.prediction !== null && Number.isFinite(candidate.prediction),
    )
    .sort(
      (left, right) =>
        right.prediction - left.prediction ||
        left.name.localeCompare(right.name),
    );
}

export function outOfRangeFeatures(
  values: Record<string, unknown>,
  ranges: FeatureRange[],
) {
  const outOfRange: Array<FeatureRange & { value: number; direction: 'below' | 'above' }> = [];
  for (const range of ranges) {
    const raw = values[range.feature];
    if (raw === undefined || raw === '') continue;
    if (typeof raw !== 'number' && typeof raw !== 'string') continue;
    const value = typeof raw === 'number' ? raw : Number(raw);
    if (!Number.isFinite(value)) continue;
    if (range.min !== null && value < range.min) {
      outOfRange.push({ ...range, value, direction: 'below' });
      continue;
    }
    if (range.max !== null && value > range.max) {
      outOfRange.push({ ...range, value, direction: 'above' });
    }
  }
  return outOfRange;
}
