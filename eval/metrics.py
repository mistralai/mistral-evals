from dataclasses import dataclass
import string


def _normalize_string(s):
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s



def _remove_end_punctuation(unnormalized_string: str) -> str:
    # remove end puncutation
    while (
        unnormalized_string
        and (
            unnormalized_string[-1] in string.punctuation
            or unnormalized_string[-1].isspace()
        )
        and unnormalized_string[-1] != "%"
    ):
        unnormalized_string = unnormalized_string[:-1]
    return unnormalized_string


@dataclass
class Metric:

    @property
    def name(self) -> str:
        raise NotImplementedError

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        raise NotImplementedError
    


class VQAMatch(Metric):
    name: str = "vqa_match"

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        if not isinstance(reference_answer, list):
            reference_answer = [reference_answer]
        normalize_response_text: str = _normalize_string(model_answer)
        matching_answers = [
            answer
            for answer in reference_answer
            if _normalize_string(answer) == normalize_response_text
        ]
        return min(1.0, float(len(matching_answers)) / 3)
    

class ANLS(Metric):

    @property
    def name(self) -> str:
        return "anls"


    def _edit_distance_helper(self, s1: str, s2: str) -> float:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = list(range(len(s1) + 1))
        for i2, c2 in enumerate(s2):
            distance_list = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distance_list.append(distances[i1])
                else:
                    distance_list.append(
                        1 + min((distances[i1], distances[i1 + 1], distance_list[-1]))
                    )
            distances = distance_list
        return distances[-1]

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        if not isinstance(reference_answer, list):
            reference_answer = [reference_answer]

        model_answer = " ".join(model_answer.strip().lower().split())
        model_answer = _remove_end_punctuation(model_answer)

        min_value = float("inf")
        for ref in reference_answer:
            # Post-processing: Normalize spaces and remove punctuations.
            ref = " ".join(ref.strip().lower().split())
            ref = _remove_end_punctuation(ref)

            # Compute edit distance
            dist = self._edit_distance_helper(ref, model_answer)
            length = max(len(ref), len(model_answer))
            value = 0.0 if length == 0 else float(dist) / float(length)
            if value < min_value:
                min_value = value

        anls_threshold = 0.0
        output = 0.0 if 1 - min_value < anls_threshold else 1 - min_value
        return output


class RelaxedCorrectness(Metric):

    @property
    def name(self) -> str:
        return "relaxed_correctness"
