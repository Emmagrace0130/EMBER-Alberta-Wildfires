import type {
  Capability,
  ContactInquiry,
  ContactInquiryResponse,
  FireMapData,
  MitigationPlan,
  Resource,
  RiskInsights,
  SearchResponse,
} from "../data/types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) {
    throw new Error(`Request to ${path} failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchPlans(): Promise<MitigationPlan[]> {
  return get<MitigationPlan[]>("/plans");
}

export function fetchPlan(id: string): Promise<MitigationPlan> {
  return get<MitigationPlan>(`/plans/${id}`);
}

export function fetchRiskInsights(): Promise<RiskInsights> {
  return get<RiskInsights>("/risk");
}

export function fetchCapabilities(): Promise<Capability[]> {
  return get<Capability[]>("/capabilities");
}

export function fetchResources(): Promise<Resource[]> {
  return get<Resource[]>("/resources");
}

export function search(query: string): Promise<SearchResponse> {
  return get<SearchResponse>(`/search?q=${encodeURIComponent(query)}`);
}

export function fetchFireMapData(): Promise<FireMapData> {
  return get<FireMapData>("/map/fires");
}

export async function submitContactInquiry(
  inquiry: ContactInquiry,
): Promise<ContactInquiryResponse> {
  const res = await fetch("/api/contact", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inquiry),
  });
  if (!res.ok) {
    throw new Error("Failed to submit contact inquiry");
  }
  return res.json() as Promise<ContactInquiryResponse>;
}
