from enum import Enum


class ModelCheckPoints(Enum):
    CODESAGE_V2_BASE = "codesage/codesage-base-v2"
    CODESAGE_V2_SMALL = "codesage/codesage-small-v2"
    CODESAGE_V2_LARGE = "codesage/codesage-large-v2"
    SFR_SMALL = "Salesforce/SFR-Embedding-Code-400M_R"


class EmbeddingModelNames(Enum):
    CODESAGE_V2_SMALL = "CodeSage-v2-Small"
    CODESAGE_V2_BASE = "CodeSage-v2-Base"
    CODESAGE_V2_LARGE = "CodeSage-v2-Large"
    SFR_EMBEDDING_SMALL = "Salesforce/SFR-Embedding-Code-400M_R"