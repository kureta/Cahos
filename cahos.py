from collections import Counter
from itertools import accumulate
from math import log2


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


class Scale:
    def __init__(self, deltas):
        self.deltas = deltas

    def __repr__(self):
        return f"{self.deltas}"

    def __str__(self):
        return f"entropy: {self.sequence_entropy:.2f}, span: {self.span}, deltas: {self.deltas}"

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

    @property
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

    def onehot_changed_voices(self):
        return [int(a != b) for a, b in zip(self.real_a, self.real_b)]

    def sequence_entropy_of_changes(self):
        return calculate_sequence_entropy(self.onehot_changed_voices())

    def common_notes(self):
        return [a for a, b in zip(self.real_a, self.real_b) if a == b]

    def changed_notes(self):
        return [(a, b) for a, b in zip(self.real_a, self.real_b) if a != b]

    def swaps(self):
        none_matches = [(a, b) for a, b in zip(self.real_a, self.real_b) if a != b]
        swaps = []
        for item in none_matches:
            if tuple(item[::-1]) in none_matches:
                swaps.append(item)

        return swaps

    def n_common_notes(self):
        return len(self.common_notes())

    def n_changed_notes(self):
        return len(self.changed_notes())

    def n_swaps(self):
        return len(self.swaps())

    def change_in_span(self):
        return self.chord_b.span - self.chord_a.span

    def n_upward_motion(self):
        n = 0
        for a, b in self.changed_notes():
            if b > a:
                n += 1
        return n

    def n_downward_motion(self):
        n = 0
        for a, b in self.changed_notes():
            if a > b:
                n += 1
        return n

    def motion_balance(self):
        if self.n_downward_motion() == 0:
            return 0.0
        ratio = self.n_upward_motion() / self.n_downward_motion()
        if ratio > 1.0:
            ratio = 1 / ratio
        return ratio

    def max_step_size(self):
        return max([abs(a - b) for a, b in self.changed_notes()])

    def n_pseudo_changes(self):
        n = 0
        for _, b in self.changed_notes():
            if b in self.real_a:
                n += 1
        return n

    def __repr__(self):
        return (
            f"seq entropy: {self.chord_a.sequence_entropy:.2f} span: {self.chord_a.span} | {self.real_a_names}\n"
            f"seq entropy: {self.chord_b.sequence_entropy:.2f} span: {self.chord_b.span} | {self.real_b_names}\n"
            f"{self.n_common_notes()} common | {self.n_changed_notes()} changes | {self.n_swaps()} swaps\n"
            f"changed places: {self.onehot_changed_voices()}\n"
            f"sequence entropy of changes: {self.sequence_entropy_of_changes():.2f}\n"
            f"max step size: {self.max_step_size()}\n"
        )
