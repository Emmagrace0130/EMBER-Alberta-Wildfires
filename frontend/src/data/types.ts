export interface MitigationPlan {
  id: string;
  title: string;
  source: string;
  summary: string;
  categories: string[];
  hazardLevel: string;
  region: string;
  jurisdiction: string;
  chunkCount: number;
  fileName?: string;
  fileSizeMb?: number;
  downloadUrl?: string;
}

export interface Resource {
  id: string;
  title: string;
  type: string;
  description: string;
  fileName?: string;
  downloadUrl?: string;
  externalNote?: string;
}

export interface FeatureImportance {
  rank: number;
  feature: string;
  importance: number;
  direction: string;
}

export interface CauseBreakdown {
  cause: string;
  count: number;
  percent: number;
}

export interface SizeClassBreakdown {
  sizeClass: string;
  label: string;
  count: number;
  percent: number;
}

export interface AnnualTrendPoint {
  year: number;
  fireCount: number;
}

export interface ModelMetric {
  model: string;
  auprc: number;
  auroc: number;
  notes: string;
}

export interface RiskInsights {
  totalFires: number;
  yearsSpanned: string;
  largeFireRate: number;
  largeFireCount: number;
  recall: number;
  precision: number;
  featureImportance: FeatureImportance[];
  causeBreakdown: CauseBreakdown[];
  sizeClassBreakdown: SizeClassBreakdown[];
  annualTrend: AnnualTrendPoint[];
  modelMetrics: ModelMetric[];
  charts: string[];
}

export interface Capability {
  id: string;
  title: string;
  description: string;
  category: string;
  status: string;
}

export interface ContactInquiry {
  name: string;
  email: string;
  organization?: string;
  inquiryType?: string;
  message: string;
}

export interface ContactInquiryResponse {
  success: boolean;
  message: string;
}

export interface SearchResultItem {
  id: string;
  type: string;
  title: string;
  snippet: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
}

export interface FireMapData {
  count: number;
  points: [number, number, number][]; // [lat, lng, intensity]
}
