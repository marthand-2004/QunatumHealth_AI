import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import { apiLogin, apiMe, type AuthUser, type TokenPair } from "../api/auth";

interface AuthContextValue {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<AuthUser>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem("access_token")
  );
  const [loading, setLoading] = useState(true);

  // On mount, restore user from localStorage or fetch from /me
  useEffect(() => {
    const stored = localStorage.getItem("auth_user");
    if (stored) {
      try {
        setUser(JSON.parse(stored) as AuthUser);
        setLoading(false);
        return;
      } catch {
        // fall through to fetch
      }
    }
    if (token) {
      apiMe()
        .then((u) => {
          setUser(u);
          localStorage.setItem("auth_user", JSON.stringify(u));
        })
        .catch(() => {
          // token invalid — clear everything
          localStorage.removeItem("access_token");
          localStorage.removeItem("refresh_token");
          localStorage.removeItem("auth_user");
          setToken(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const persistTokens = (pair: TokenPair) => {
    localStorage.setItem("access_token", pair.access_token);
    localStorage.setItem("refresh_token", pair.refresh_token);
    setToken(pair.access_token);
  };

  const login = useCallback(async (email: string, password: string): Promise<AuthUser> => {
    const pair = await apiLogin(email, password);
    persistTokens(pair);
    const me = await apiMe();
    setUser(me);
    localStorage.setItem("auth_user", JSON.stringify(me));
    return me;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("auth_user");
    setUser(null);
    setToken(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuthContext(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuthContext must be used inside <AuthProvider>");
  return ctx;
}
