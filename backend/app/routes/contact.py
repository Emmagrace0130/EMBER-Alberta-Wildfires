from fastapi import APIRouter

from ..schemas import ContactInquiry, ContactInquiryResponse
from ..services import contact_service

router = APIRouter(prefix="/api/contact", tags=["contact"])


@router.post("", response_model=ContactInquiryResponse)
def submit_contact_inquiry(inquiry: ContactInquiry) -> ContactInquiryResponse:
    contact_service.save_contact_inquiry(inquiry)
    return ContactInquiryResponse(
        success=True,
        message="Thanks for reaching out — the EMBER team will follow up soon.",
    )
