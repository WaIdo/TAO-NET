"""Semantic-enhanced prompt templates (SPK) for labeling OOD traffic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SPKTemplate:
    """Container describing prompt modes (strict/complete/extended)."""

    name: str
    mode: str
    description: str
    candidate_labels: List[str]
    knowledge_hints: List[str]

    def instruction_block(self) -> str:
        labels = ", ".join(self.candidate_labels)
        hints = "\n".join(f"- {hint}" for hint in self.knowledge_hints)
        return (
            f"Mode: {self.mode}\n"
            f"Objective: {self.description}\n"
            f"Candidate labels: {labels}\n"
            f"Domain hints:\n{hints}\n"
            "Respond ONLY with a JSON object {\"label\": <label>, \"rationale\": <short reason>} "
            "where <label> must come from the candidate list."
        )


_EXTENDED_TEMPLATE = SPKTemplate(
    name="Cross-domain Extended",
    mode="extended",
    description="Generalize across CHNAPP + VPN + Tor corpora",
    candidate_labels=[
        "WeChat",
        "Weibo",
        "YouTube",
        "Spotify",
        "VoipBuster",
        "Vimeo",
        "Thunderbird",
        "SSL",
    ],
    knowledge_hints=[
        "VoipBuster voice calls show steady packet pacing with symmetric directions",
        "Tor/SSL tunnels have uniform record sizes compared with consumer apps",
        "Streaming apps (YouTube, Vimeo, Spotify) exhibit long downstream-dominant bursts",
    ],
)


_DEFAULT_LABEL_SPACES: Dict[str, Dict[str, SPKTemplate]] = {
    "Tinghuaall": {
        "strict": SPKTemplate(
            name="Tinghua Strict",
            mode="strict",
            description="Disambiguate between newly emerging mobile apps detected as OOD",
            candidate_labels=["WeChat", "Weibo"],
            knowledge_hints=[
                "WeChat traffic often shows bidirectional short bursts with TLS records mirroring mobile messaging",
                "Weibo traces contain longer downstream payloads triggered by media timelines",
            ],
        ),
        "complete": SPKTemplate(
            name="Tinghua Complete",
            mode="complete",
            description="Choose the most likely application from the combined ID+OOD taxonomy",
            candidate_labels=["QQMail", "QQMusic", "Youku", "TaoBao", "WeChat", "Weibo"],
            knowledge_hints=[
                "QQMail and QQMusic are service-specific with periodic keep-alives",
                "Youku streams longer video segments with consistent chunk sizes",
                "TaoBao e-commerce traces show short TLS handshakes followed by bursts of JSON payloads",
            ],
        ),
        "extended": _EXTENDED_TEMPLATE,
    },
    "VPN": {
        "strict": SPKTemplate(
            name="VPN Strict",
            mode="strict",
            description="Identify VPN-encrypted OOD consumer apps",
            candidate_labels=["VoipBuster", "YouTube", "Vimeo", "Spotify"],
            knowledge_hints=[
                "VoipBuster and Spotify are audio-centric with regular packet intervals",
                "YouTube/Vimeo deliver adaptive video chunks with large downstream frames",
            ],
        ),
        "complete": SPKTemplate(
            name="VPN Complete",
            mode="complete",
            description="Classify VPN flows among all observed services",
            candidate_labels=[
                "Gmail",
                "Facebook",
                "FTPS",
                "Hangouts",
                "Netflix",
                "BitTorrent",
                "SFTP",
                "Skype",
                "VoipBuster",
                "YouTube",
                "Vimeo",
                "Spotify",
            ],
            knowledge_hints=[
                "Enterprise tools (Gmail, FTPS, SFTP) keep consistent record sizes",
                "Consumer media (Netflix, YouTube, Spotify) have bursty downstream traffic",
                "BitTorrent shows multi-peer uplink spikes",
            ],
        ),
        "extended": _EXTENDED_TEMPLATE,
    },
    "Nontor": {
        "strict": SPKTemplate(
            name="Tor Strict",
            mode="strict",
            description="Differentiate Tor OOD services",
            candidate_labels=["SSL", "Thunderbird", "Vimeo", "YouTube"],
            knowledge_hints=[
                "Thunderbird mail uses SMTP/IMAP patterns even over Tor",
                "Video services on Tor show longer contiguous TLS records",
            ],
        ),
        "complete": SPKTemplate(
            name="Tor Complete",
            mode="complete",
            description="Classify Tor flows across ID + OOD services",
            candidate_labels=[
                "Gmail",
                "Facebook",
                "FTP",
                "Hangout",
                "P2P",
                "POP",
                "Skype",
                "Spotify",
                "SSL",
                "Thunderbird",
                "Vimeo",
                "YouTube",
            ],
            knowledge_hints=[
                "POP/SMTP exhibit command-response cycles",
                "P2P traffic features bidirectional symmetry and irregular packet sizes",
            ],
        ),
        "extended": _EXTENDED_TEMPLATE,
    },
}


class SemanticEnhancedPromptKnowledge:
    """Dynamic repository that stores dataset-specific SPK templates."""

    def __init__(self, dataset_name: str):
        if dataset_name not in _DEFAULT_LABEL_SPACES:
            raise KeyError(f"Unsupported dataset '{dataset_name}' for SPK")
        self.dataset_name = dataset_name
        self._templates = _DEFAULT_LABEL_SPACES[dataset_name]

    def available_modes(self) -> List[str]:
        return list(self._templates.keys())

    def get(self, mode: str) -> SPKTemplate:
        if mode not in self._templates:
            raise KeyError(f"Mode '{mode}' is not defined for dataset '{self.dataset_name}'")
        return self._templates[mode]
