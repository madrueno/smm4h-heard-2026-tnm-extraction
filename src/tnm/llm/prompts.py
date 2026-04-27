"""Stage-specific prompts for TNM classification using AJCC criteria."""

from abc import ABC, abstractmethod


class StagePrompt(ABC):
    @abstractmethod
    def get_prompt(self, text: str) -> str: ...


class TStagePrompt(StagePrompt):
    TEMPLATE = (
        'You are an expert oncology pathologist.\n\n'
        'Based on the AJCC criteria, classify the T stage from the pathology report:\n'
        '- T1: Tumor with limited size or extent\n'
        '- T2: Tumor with greater size or local extent\n'
        '- T3: Tumor with more advanced local extension\n'
        '- T4: Tumor with the most extensive local invasion\n\n'
        'Pathology Report:\n{text}\n\n'
        'Answer with ONLY one label: T1, T2, T3, or T4.'
    )

    def get_prompt(self, text: str) -> str:
        return self.TEMPLATE.format(text=text)


class NStagePrompt(StagePrompt):
    TEMPLATE = (
        'You are an expert oncology pathologist.\n\n'
        'Based on the AJCC criteria, classify the N stage from the pathology report:\n'
        '- N0: No regional lymph node involvement\n'
        '- N1: Mild regional lymph node involvement\n'
        '- N2: Moderate regional lymph node involvement\n'
        '- N3: Extensive regional lymph node involvement\n\n'
        'Pathology Report:\n{text}\n\n'
        'Answer with ONLY one label: N0, N1, N2, or N3.'
    )

    def get_prompt(self, text: str) -> str:
        return self.TEMPLATE.format(text=text)


class MStagePrompt(StagePrompt):
    TEMPLATE = (
        'You are an expert oncology pathologist.\n\n'
        'Based on the AJCC criteria, classify the M stage from the pathology report:\n'
        '- M0: No distant metastasis identified.\n'
        '- M1: Distant metastasis confirmed.\n\n'
        'Pathology Report:\n{text}\n\n'
        'Answer with ONLY one label: M0 or M1.'
    )

    def get_prompt(self, text: str) -> str:
        return self.TEMPLATE.format(text=text)


if __name__ == '__main__':
    sample_text = (
        'The tumor is 3 cm in size and has invaded the surrounding tissue. '
        'No lymph node involvement is observed. '
        'No distant metastases are detected.'
    )
    print(MStagePrompt().get_prompt(sample_text))
