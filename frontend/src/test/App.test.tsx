import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, it, expect } from "vitest";
import { AuthProvider } from "../context/AuthContext";
import LoginPage from "../pages/LoginPage";

function Wrapper({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <MemoryRouter>{children}</MemoryRouter>
    </AuthProvider>
  );
}

describe("LoginPage", () => {
  it("renders sign-in heading", () => {
    render(<LoginPage />, { wrapper: Wrapper });
    expect(screen.getByRole("heading", { name: /sign in/i })).toBeInTheDocument();
  });

  it("renders email and password fields", () => {
    render(<LoginPage />, { wrapper: Wrapper });
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  });

  it("renders register link", () => {
    render(<LoginPage />, { wrapper: Wrapper });
    expect(screen.getByRole("link", { name: /register/i })).toBeInTheDocument();
  });
});
