import { Github, Linkedin, FileText, Mail } from "lucide-react";
import logo from "@/assets/logo.png";

const Footer = () => {
  const links = [
    { icon: Github, label: "GitHub", url: "https://github.com/BBEK-Anand/PyTorchLabFlow" },
    { icon: FileText, label: "Medium Blog", url: "https://medium.com/@bbek-anand" },
    { icon: Linkedin, label: "LinkedIn", url: "https://linkedin.com/in/bbek-anand" },
    { icon: Mail, label: "Email", url: "mailto:bbek-anand@example.com" }
  ];

  return (
    <footer className="border-t border-primary/20 py-12 px-4">
      <div className="container max-w-7xl mx-auto">
        <div className="grid md:grid-cols-3 gap-12 mb-8">
          {/* Brand */}
          <div className="space-y-4">
            <img src={logo} alt="PyTorchLabFlow" className="h-16 w-auto" />
            <p className="text-sm text-muted-foreground">
              Research-first experiment tracking for deep learning. Built by researchers, for researchers.
            </p>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Resources</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
                  Documentation
                </a>
              </li>
              <li>
                <a href="https://github.com/BBEK-Anand/PyTorchLabFlow" className="text-muted-foreground hover:text-primary transition-colors">
                  GitHub Repository
                </a>
              </li>
              <li>
                <a href="https://github.com/BBEK-Anand/Military_AirCraft_Classification" className="text-muted-foreground hover:text-primary transition-colors">
                  Example Project
                </a>
              </li>
              <li>
                <a href="https://medium.com/@bbek-anand/why-i-built-pytorchlabflow-research-first-experiment-tracking-for-deep-learning-chaos-a50b5bc47dde" className="text-muted-foreground hover:text-primary transition-colors">
                  Why PyTorchLabFlow?
                </a>
              </li>
            </ul>
          </div>

          {/* Connect */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Connect</h3>
            <div className="flex flex-wrap gap-3">
              {links.map((link, index) => (
                <a
                  key={index}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 rounded-lg bg-primary/10 hover:bg-primary/20 flex items-center justify-center transition-colors"
                  aria-label={link.label}
                >
                  <link.icon className="h-5 w-5 text-primary" />
                </a>
              ))}
            </div>
            <p className="text-sm text-muted-foreground">
              Have questions or want to collaborate? Reach out anytime.
            </p>
          </div>
        </div>

        {/* Bottom */}
        <div className="pt-8 border-t border-primary/10 flex flex-wrap justify-between items-center gap-4 text-sm text-muted-foreground">
          <p>© 2025 PyTorchLabFlow. Licensed under Apache 2.0</p>
          <p>
            Built by{" "}
            <a href="https://github.com/BBEK-Anand" className="text-primary hover:text-accent transition-colors">
              BBEK-Anand
            </a>
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
