import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import About from "./pages/About";
import MitigationPlans from "./pages/MitigationPlans";
import MitigationPlanDetail from "./pages/MitigationPlanDetail";
import RiskInsights from "./pages/RiskInsights";
import Capabilities from "./pages/Capabilities";
import HowItWorks from "./pages/HowItWorks";
import Resources from "./pages/Resources";
import Contact from "./pages/Contact";
import FireMap from "./pages/FireMap";
import Chat from "./pages/Chat";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/mitigation-plans" element={<MitigationPlans />} />
        <Route path="/mitigation-plans/:id" element={<MitigationPlanDetail />} />
        <Route path="/risk-insights" element={<RiskInsights />} />
        <Route path="/what-ember-can-do" element={<Capabilities />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/fire-map" element={<FireMap />} />
        <Route path="/assistant" element={<Chat />} />
      </Routes>
    </Layout>
  );
}
