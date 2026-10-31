import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toVideoIds } from '../src/lib/videoIds';

test('topics without a video render no player', () => {
  assert.deepEqual(toVideoIds(null), []);
  assert.deepEqual(toVideoIds(undefined), []);
  assert.deepEqual(toVideoIds(''), []);
  assert.deepEqual(toVideoIds([]), []);
});

test('a single-part topic yields a list of one', () => {
  assert.deepEqual(toVideoIds('abc123def45'), ['abc123def45']);
});

test('a multi-part topic keeps its parts in order', () => {
  assert.deepEqual(toVideoIds(['xyz789ghi01', 'abc123def45']), ['xyz789ghi01', 'abc123def45']);
});
