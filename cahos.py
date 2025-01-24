from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
from math import log2
from typing import Annotated, List, Literal, Sequence, Tuple, TypeAlias, cast

import polars as pl

IntSequence11: TypeAlias = Annotated[Sequence[int], 11]
IntSequence12: TypeAlias = Annotated[Sequence[int], 12]
StrSequence12: TypeAlias = Annotated[Sequence[str], 12]
BinarySequence12: TypeAlias = Annotated[Sequence[Literal[0, 1]], 12]
NotePair: TypeAlias = Tuple[int, int]


def calculate_entropy(sequence: Sequence) -> float:
    # Count the frequency of each element in the sequence
    frequency = Counter(sequence)
    total_count = len(sequence)

    # Calculate the entropy
    entropy = 0
    for count in frequency.values():
        probability = count / total_count
        entropy -= probability * log2(probability)

    return entropy


def calculate_subseq_entropy(sequence: Sequence, subsequence_length: int) -> float:
    sub_seqs = []
    for idx in range(len(sequence) - subsequence_length + 1):
        sub_seqs.append(tuple(sequence[idx : idx + subsequence_length]))
    return calculate_entropy(sub_seqs)


def calculate_sequence_entropy(sequence):
    return sum(
        calculate_subseq_entropy(sequence, length)
        for length in range(len(sequence) + 1)
    )


@dataclass
class Scale:
    intervals: IntSequence11
    pitch_classes: IntSequence12
    span: int
    interval_set: List[int]
    n_unique_intervals: int
    entropy: float
    bigram_entropy: float
    sequence_entropy: float


scale_schema = {
    "intervals": pl.List(pl.UInt8),
    "pitch_classes": pl.List(pl.UInt8),
    "span": pl.UInt8,
    "interval_set": pl.List(pl.UInt8),
    "n_unique_intervals": pl.UInt8,
    "entropy": pl.Float64,
    "bigram_entropy": pl.Float64,
    "sequence_entropy": pl.Float64,
}


def make_scale(intervals: IntSequence11) -> Scale:
    return Scale(
        intervals=intervals,
        pitch_classes=[a % 12 for a in accumulate(intervals, initial=0)],
        span=sum(intervals),
        interval_set=list(set(intervals)),
        n_unique_intervals=len(set(intervals)),
        entropy=calculate_entropy(intervals),
        bigram_entropy=calculate_subseq_entropy(intervals, 2),
        sequence_entropy=calculate_sequence_entropy(intervals),
    )


def contains_subsequence(main_list: Sequence[int], sub_list: Sequence[int]) -> bool:
    for idx in range(len(main_list) - len(sub_list) + 1):
        if main_list[idx : idx + len(sub_list)] == sub_list:
            return True
    return False


def get_scales(
    allowed_intervals: Sequence[int],
    disallowed_subsequences: Sequence[Sequence[int]] = [],
    disallowed_beginnings: Sequence[int] = [],
    max_span: int = 88,
) -> List[Scale]:
    result = []
    get_scales_recursive(
        allowed_intervals,
        disallowed_subsequences,
        disallowed_beginnings,
        max_span,
        result,
    )

    return result


def get_scales_recursive(
    allowed_intervals: Sequence[int],
    disallowed_subsequences: Sequence[Sequence[int]],
    disallowed_beginnings: Sequence[int],
    max_span: int,
    deltas_accumulator: List[Scale],
    pitch_classes: List[int] = [0],
    deltas: List[int] = [],
    idx: int = 0,
):
    if idx == 11:
        deltas_accumulator.append(make_scale(deltas))
        # normal exit
        return
    for next_interval in allowed_intervals:
        # early exits
        if (idx == 0) and (next_interval in disallowed_beginnings):
            continue
        if sum(deltas + [next_interval]) > max_span:
            continue
        if any(
            contains_subsequence(deltas + [next_interval], dis_seq)
            for dis_seq in disallowed_subsequences
        ):
            continue
        next_note = (pitch_classes[-1] + next_interval) % 12
        if next_note not in pitch_classes:
            get_scales_recursive(
                allowed_intervals,
                disallowed_subsequences,
                disallowed_beginnings,
                max_span,
                deltas_accumulator,
                pitch_classes + [next_note],
                deltas + [next_interval],
                idx + 1,
            )


def n_swaps(xs: Sequence[int], ys: Sequence[int]) -> int:
    none_matches = [(a, b) for a, b in zip(xs, ys) if a != b]
    swaps = []
    for item in none_matches:
        if tuple(item[::-1]) in none_matches:
            swaps.append(item)

    return len(swaps)


def mark_changes(first: Sequence[int], second: Sequence[int]) -> BinarySequence12:
    return cast(BinarySequence12, [int(a != b) for a, b in zip(first, second)])


note_names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_note_name(midi_number: int) -> str:
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12
    return f"{note_names[note_index]}{octave}"


@dataclass
class VoiceLeading:
    intervals_a: IntSequence11
    intervals_b: IntSequence11
    base_a: int
    base_b: int
    midis_a: IntSequence12
    midis_b: IntSequence12
    note_names_a: StrSequence12
    note_names_b: StrSequence12
    onehot_changed_voices: BinarySequence12
    entropy_of_changes: float
    sequence_entropy_of_changes: float
    common_notes: Sequence[int]
    changed_notes: Sequence[NotePair]
    swaps: Sequence[NotePair]
    n_changed_notes: int
    n_common_notes: int
    n_swaps: int
    change_in_span: int
    n_upward_motion: int
    n_downward_motion: int
    motion_balance: float
    max_step_size: int
    n_pseudo_changes: int


voice_leading_schema = {
    "intervals_a": pl.List(pl.UInt8),
    "intervals_b": pl.List(pl.UInt8),
    "base_a": pl.UInt8,
    "base_b": pl.UInt8,
    "midis_a": pl.List(pl.UInt8),
    "midis_b": pl.List(pl.UInt8),
    "note_names_a": pl.List(pl.String),
    "note_names_b": pl.List(pl.String),
    "onehot_changed_voices": pl.List(pl.Binary),
    "entropy_of_changes": pl.Float64,
    "sequence_entropy_of_changes": pl.Float64,
    "common_notes": pl.List(pl.UInt8),
    "changed_notes": pl.List(pl.List(pl.UInt8)),
    "swaps": pl.List(pl.List(pl.UInt8)),
    "n_changed_notes": pl.UInt8,
    "n_common_notes": pl.UInt8,
    "n_swaps": pl.UInt8,
    "change_in_span": pl.Int8,
    "n_upward_motion": pl.UInt8,
    "n_downward_motion": pl.UInt8,
    "motion_balance": pl.Float64,
    "max_step_size": pl.UInt8,
    "n_pseudo_changes": pl.UInt8,
}


def intervals_to_midis(intervals: Sequence[int], start: int) -> List[int]:
    return [a for a in accumulate(intervals, initial=start)]


def midis_to_names(midis: Sequence[int]) -> List[str]:
    return [midi_to_note_name(x) for x in midis]


def get_swaps(midis_a: Sequence[int], midis_b: Sequence[int]) -> List[NotePair]:
    none_matches = [(a, b) for a, b in zip(midis_a, midis_b) if a != b]
    swaps = []
    for item in none_matches:
        if tuple(item[::-1]) in none_matches:
            swaps.append(item)

    return swaps


def get_span(midis: Sequence[int]) -> int:
    return midis[-1] - midis[0]


def count_upward_motion(changed_notes: Sequence[NotePair]) -> int:
    n = 0
    for a, b in changed_notes:
        if b > a:
            n += 1
    return n


def count_downward_motion(changed_notes: Sequence[NotePair]) -> int:
    n = 0
    for a, b in changed_notes:
        if a > b:
            n += 1
    return n


def calculate_motion_balance(n_upward_motion: int, n_downward_motion: int) -> float:
    if n_downward_motion == 0:
        return 0.0
    ratio = n_upward_motion / n_downward_motion
    if ratio > 1.0:
        ratio = 1 / ratio
    return ratio


def get_max_step_size(changed_notes: Sequence[NotePair]) -> int:
    return max([abs(a - b) for a, b in changed_notes])


def get_n_pseudo_changes(
    midis_a: Sequence[int], changed_notes: Sequence[NotePair]
) -> int:
    n = 0
    for _, b in changed_notes:
        if b in midis_a:
            n += 1
    return n


def make_voice_leading(
    intervals_a: IntSequence11, base_a: int, intervals_b: IntSequence11, base_b: int
) -> VoiceLeading:
    midis_a = intervals_to_midis(intervals_a, base_a)
    midis_b = intervals_to_midis(intervals_b, base_b)
    changes = mark_changes(midis_a, midis_b)
    common_notes = [a for a, b in zip(midis_a, midis_b) if a == b]
    changed_notes = [(a, b) for a, b in zip(midis_a, midis_b) if a != b]
    swaps = get_swaps(midis_a, midis_b)
    change_in_span = get_span(midis_b) - get_span(midis_a)
    n_upward_motion = count_upward_motion(changed_notes)
    n_downward_motion = count_downward_motion(changed_notes)

    return VoiceLeading(
        intervals_a=intervals_a,
        intervals_b=intervals_b,
        base_a=base_a,
        base_b=base_b,
        midis_a=midis_a,
        midis_b=midis_b,
        note_names_a=midis_to_names(midis_a),
        note_names_b=midis_to_names(midis_b),
        onehot_changed_voices=changes,
        entropy_of_changes=calculate_entropy(changes),
        sequence_entropy_of_changes=calculate_sequence_entropy(changes),
        common_notes=common_notes,
        changed_notes=changed_notes,
        swaps=swaps,
        n_changed_notes=len(changed_notes),
        n_common_notes=len(common_notes),
        n_swaps=len(swaps),
        change_in_span=change_in_span,
        n_upward_motion=n_upward_motion,
        n_downward_motion=n_downward_motion,
        motion_balance=calculate_motion_balance(n_upward_motion, n_downward_motion),
        max_step_size=get_max_step_size(changed_notes),
        n_pseudo_changes=get_n_pseudo_changes(midis_a, changed_notes),
    )
