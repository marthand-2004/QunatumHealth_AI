import { Navigate } from "react-router-dom";
import { useAuthContext } from "../context/AuthContext";

interface Props {
  children: React.ReactNode;
  roles: string[];
}

const ROLE_HOME: Record<string, string> = {
  patient: "/dashboard",
  doctor: "/clinical",
  admin: "/admin",
};

export default function ProtectedRoute({ children, roles }: Props) {
  const { user, loading } = useAuthContext();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <span className="text-gray-500 text-sm">Loading…</span>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;

  if (!roles.includes(user.role)) {
    // Redirect to the user's own home page instead of /login
    return <Navigate to={ROLE_HOME[user.role] ?? "/login"} replace />;
  }

  return <>{children}</>;
}
