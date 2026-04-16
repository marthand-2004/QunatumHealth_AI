import axios from "axios";

export interface AuthUser {
  id: string;
  email: string;
  role: "patient" | "doctor" | "admin";
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

const api = axios.create({ baseURL: "/api" });

// Attach access token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// On 401, attempt token refresh once then redirect to login
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config;
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true;
      const refreshToken = localStorage.getItem("refresh_token");
      if (refreshToken) {
        try {
          const { data } = await axios.post<TokenPair>("/api/auth/refresh", {
            refresh_token: refreshToken,
          });
          localStorage.setItem("access_token", data.access_token);
          localStorage.setItem("refresh_token", data.refresh_token);
          original.headers.Authorization = `Bearer ${data.access_token}`;
          return api(original);
        } catch {
          // refresh failed — clear storage and redirect
        }
      }
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      localStorage.removeItem("auth_user");
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

export async function apiLogin(email: string, password: string): Promise<TokenPair> {
  const { data } = await api.post<TokenPair>("/auth/login", { email, password });
  return data;
}

export async function apiRegister(
  email: string,
  password: string,
  role: "patient" | "doctor" | "admin"
): Promise<{ message: string }> {
  const { data } = await api.post<{ message: string }>("/auth/register", {
    email,
    password,
    role,
  });
  return data;
}

export async function apiRefresh(refreshToken: string): Promise<TokenPair> {
  const { data } = await api.post<TokenPair>("/auth/refresh", {
    refresh_token: refreshToken,
  });
  return data;
}

export async function apiMe(): Promise<AuthUser> {
  const { data } = await api.get<AuthUser>("/auth/me");
  return data;
}

export default api;
