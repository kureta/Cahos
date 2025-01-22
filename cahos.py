from collections import Counter
from dataclasses import asdict, dataclass
from itertools import accumulate
from math import log2
from typing import List


def dataclass_to_dict_with_properties(obj):
    # Start with regular fields
    data = asdict(obj)
    # Add properties
    for attr_name in dir(obj):
        if isinstance(getattr(type(obj), attr_name, None), property):
            data[attr_name] = getattr(obj, attr_name)
    return data


def calculate_entropy(sequence):
    # Count the frequency of each element in the sequence
    frequency = Counter(sequence)
    total_count = len(sequence)

    # Calculate the entropy
    entropy = 0
    for count in frequency.values():
        probability = count / total_count
        entropy -= probability * log2(probability)

    return entropy


def calculate_subseq_entropy(sequence, subsequence_length):
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
    deltas: List[int]

    @property
    def pitch_classes(self):
        return [a % 12 for a in accumulate(self.deltas, initial=0)]

    def transposed(self, start=0):
        return [a % 12 for a in accumulate(self.deltas, initial=start)]

    def realization(self, start=0):
        return [a for a in accumulate(self.deltas, initial=start)]

    @property
    def span(self):
        return sum(self.deltas)

    @property
    def intervals(self):
        return list(set(self.deltas))

    @property
    def n_intervals(self):
        return len(set(self.deltas))

    @property
    def entropy(self):
        return calculate_entropy(self.deltas)

    def transition_entropy(self):
        return calculate_subseq_entropy(self.deltas, 2)

    @property
    def sequence_entropy(self):
        return calculate_sequence_entropy(self.deltas)


def contains_subsequence(main_list, sub_list):
    for idx in range(len(main_list) - len(sub_list) + 1):
        if main_list[idx : idx + len(sub_list)] == sub_list:
            return True
    return False


def get_scales(
    allowed_intervals, disallowed_subsequences=[], disallowed_beginnings=[], max_span=88
):
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
    allowed_intervals,
    disallowed_subsequences,
    disallowed_beginnings,
    max_span,
    deltas_accumulator,
    pitch_classes=[0],
    deltas=[],
    idx=0,
):
    if idx == 11:
        deltas_accumulator.append(deltas)
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


def n_swaps(xs, ys):
    none_matches = [(a, b) for a, b in zip(xs, ys) if a != b]
    swaps = []
    for item in none_matches:
        if tuple(item[::-1]) in none_matches:
            swaps.append(item)

    return len(swaps)


def mark_changes(first, second):
    return [int(a != b) for a, b in zip(first, second)]


note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_number):
    octave = (midi_number // 12) - 1
    note_index = midi_number % 12
    return f"{note_names[note_index]}{octave}"


@dataclass
class VoiceLeading:
    def __init__(self, chord_a: Scale, base_a: int, chord_b: Scale, base_b: int):
        self.chord_a = chord_a
        self.base_a = base_a
        self.chord_b = chord_b
        self.base_b = base_b

    @property
    def real_a(self):
        return self.chord_a.realization(self.base_a)

    @property
    def real_b(self):
        return self.chord_b.realization(self.base_b)

    @property
    def real_a_names(self):
        return [midi_to_note_name(x) for x in self.chord_a.realization(self.base_a)]

    @property
    def real_b_names(self):
        return [midi_to_note_name(x) for x in self.chord_b.realization(self.base_b)]

    @property
    def onehot_changed_voices(self):
        return [int(a != b) for a, b in zip(self.real_a, self.real_b)]

    @property
    def sequence_entropy_of_changes(self):
        return calculate_sequence_entropy(self.onehot_changed_voices)

    @property
    def common_notes(self):
        return [a for a, b in zip(self.real_a, self.real_b) if a == b]

    @property
    def changed_notes(self):
        return [(a, b) for a, b in zip(self.real_a, self.real_b) if a != b]

    @property
    def swaps(self):
        none_matches = [(a, b) for a, b in zip(self.real_a, self.real_b) if a != b]
        swaps = []
        for item in none_matches:
            if tuple(item[::-1]) in none_matches:
                swaps.append(item)

        return swaps

    @property
    def n_common_notes(self):
        return len(self.common_notes)

    @property
    def n_changed_notes(self):
        return len(self.changed_notes)

    @property
    def n_swaps(self):
        return len(self.swaps)

    @property
    def change_in_span(self):
        return self.chord_b.span - self.chord_a.span

    @property
    def n_upward_motion(self):
        n = 0
        for a, b in self.changed_notes:
            if b > a:
                n += 1
        return n

    @property
    def n_downward_motion(self):
        n = 0
        for a, b in self.changed_notes:
            if a > b:
                n += 1
        return n

    @property
    def motion_balance(self):
        if self.n_downward_motion == 0:
            return 0.0
        ratio = self.n_upward_motion / self.n_downward_motion
        if ratio > 1.0:
            ratio = 1 / ratio
        return ratio

    @property
    def max_step_size(self):
        return max([abs(a - b) for a, b in self.changed_notes])

    @property
    def n_pseudo_changes(self):
        n = 0
        for _, b in self.changed_notes:
            if b in self.real_a:
                n += 1
        return n
