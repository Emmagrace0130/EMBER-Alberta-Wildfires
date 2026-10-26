from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class MitigationPlan(BaseModel):
    id: str
    title: str
    source: str
    summary: str
    categories: list[str]
    hazardLevel: str
    region: str
    jurisdiction: str = "Alberta"
    chunkCount: int
    fileName: Optional[str] = None
    fileSizeMb: Optional[float] = None
    downloadUrl: Optional[str] = None


class Resource(BaseModel):
    id: str
    title: str
    type: str
    description: str
    fileName: Optional[str] = None
    downloadUrl: Optional[str] = None
    externalNote: Optional[str] = None


class FeatureImportance(BaseModel):
    rank: int
    feature: str
    importance: float
    direction: str


class CauseBreakdown(BaseModel):
    cause: str
    count: int
    percent: float


class SizeClassBreakdown(BaseModel):
    sizeClass: str
    label: str
    count: int
    percent: float


class AnnualTrendPoint(BaseModel):
    year: int
    fireCount: int


class ModelMetric(BaseModel):
    model: str
    auprc: float
    auroc: float
    notes: str


class RiskInsights(BaseModel):
    totalFires: int
    yearsSpanned: str
    largeFireRate: float
    largeFireCount: int
    recall: float
    precision: float
    featureImportance: list[FeatureImportance]
    causeBreakdown: list[CauseBreakdown]
    sizeClassBreakdown: list[SizeClassBreakdown]
    annualTrend: list[AnnualTrendPoint]
    modelMetrics: list[ModelMetric]
    charts: list[str]


class Capability(BaseModel):
    id: str
    title: str
    description: str
    category: str
    status: str


class ContactInquiry(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    organization: Optional[str] = None
    inquiryType: Optional[str] = None
    message: str = Field(min_length=1, max_length=5000)


class ContactInquiryResponse(BaseModel):
    success: bool
    message: str


class SearchResultItem(BaseModel):
    id: str
    type: str
    title: str
    snippet: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
