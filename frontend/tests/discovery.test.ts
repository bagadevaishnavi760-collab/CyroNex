import assert from 'node:assert/strict';
import test from 'node:test';
import {
  candidateCompleteness,
  orderedFeaturePayload,
  outOfRangeFeatures,
  rankCandidates,
} from '../src/discovery.ts';

const features = Array.from({ length: 81 }, (_, index) => `feature_${index}`);
const completeValues = Object.fromEntries(features.map((feature, index) => [feature, index]));

test('ordered payload preserves all model features and excludes dataset target fields', () => {
  const payload = orderedFeaturePayload(
    { ...completeValues, critical_temp: 99 },
    features,
  );
  assert.deepEqual(Object.keys(payload), features);
  assert.equal('critical_temp' in payload, false);
});

test('candidate completeness reports missing and invalid values', () => {
  const missing = { ...completeValues };
  delete missing[features[4]];
  missing[features[5]] = '';
  assert.equal(candidateCompleteness(missing, features).complete, false);
  assert.deepEqual(candidateCompleteness(missing, features).missing, [
    features[4],
    features[5],
  ]);

  const invalid = { ...completeValues, [features[7]]: 'NaN' };
  assert.deepEqual(candidateCompleteness(invalid, features).invalid, [features[7]]);
  assert.equal(candidateCompleteness(invalid, features).supplied, 80);
  assert.throws(() => orderedFeaturePayload(invalid, features), /feature_7/);
  const booleanValue = { ...completeValues, [features[8]]: true };
  assert.deepEqual(candidateCompleteness(booleanValue, features).invalid, [features[8]]);
  assert.throws(() => orderedFeaturePayload(booleanValue, features), /feature_8/);
  assert.throws(
    () => orderedFeaturePayload(completeValues, features.slice(1)),
    /exactly 81 features/,
  );
});

test('candidate ranking sorts successful estimates and omits unpredicted candidates', () => {
  const ranked = rankCandidates([
    { id: 'a', name: 'Lower', prediction: 12 },
    { id: 'b', name: 'Not run', prediction: null },
    { id: 'c', name: 'Higher', prediction: 31 },
    { id: 'd', name: 'Tie first', prediction: 31 },
  ]);
  assert.deepEqual(ranked.map((candidate) => candidate.id), ['c', 'd', 'a']);
});

test('range checks use the observed feature minimum and maximum', () => {
  assert.deepEqual(
    outOfRangeFeatures(
      { low: -1, normal: 5, high: 11 },
      [
        { feature: 'low', min: 0, max: 10 },
        { feature: 'normal', min: 0, max: 10 },
        { feature: 'high', min: 0, max: 10 },
      ],
    ).map(({ feature, direction }) => [feature, direction]),
    [['low', 'below'], ['high', 'above']],
  );
});
