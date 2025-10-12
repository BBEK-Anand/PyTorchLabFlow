import Hero from "@/components/Hero";
import Introduction from "@/components/Introduction";
import TheProblem from "@/components/TheProblem";
import HowItWorks from "@/components/HowItWorks";
import Installation from "@/components/Installation";
import CallToAction from "@/components/CallToAction";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Hero />
      <Introduction />
      <TheProblem />
      <HowItWorks />
      <Installation />
      <CallToAction />
      <Footer />
    </div>
  );
};

export default Index;
