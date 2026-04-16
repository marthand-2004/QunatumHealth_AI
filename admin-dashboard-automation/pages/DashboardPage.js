// DashboardPage.js
// Page Object Model for the Owlet Campus authenticated admin dashboard.
// Encapsulates post-login state: sidebar navigation and user identity display.
// Locators use semantic Playwright APIs — no CSS selectors — for resilience.
// Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 1.4, 1.6

const { expect } = require('@playwright/test');

class DashboardPage {
  /** @param {import('@playwright/test').Page} page */
  constructor(page) {
    this.page = page;
  }

  // ---------------------------------------------------------------------------
  // Locator accessors
  // Getters return a fresh Playwright Locator each time they are called.
  // Playwright locator errors (e.g. TimeoutError) are intentionally not caught
  // here — they propagate to the caller so failures are visible in test output.
  // ---------------------------------------------------------------------------

  /**
   * Sidebar navigation — the <nav> landmark in the admin dashboard.
   * DOM snapshot confirms: navigation [ref=e8] with sidebar links.
   */
  get sidebarNav() {
    return this.page.getByRole('navigation');
  }

  /**
   * Top-right user identity — the "Account menu for raghuram" button in the banner.
   * DOM snapshot confirms: button "Account menu for raghuram" inside banner [ref=e66].
   */
  get userIdentity() {
    return this.page.getByRole('banner').getByRole('button', { name: /account menu/i });
  }

  // ---------------------------------------------------------------------------
  // waitForLoad() — Requirement 3.3
  // Waits until the sidebar navigation is visible, confirming the dashboard
  // has fully loaded after a successful login.
  // ---------------------------------------------------------------------------
  async waitForLoad() {
    // Wait up to 30s for the sidebar nav — the SPA may take time after login redirect
    await expect(this.sidebarNav).toBeVisible({ timeout: 30000 });
  }

  // ---------------------------------------------------------------------------
  // navigateTo(sectionName) — Requirement 3.4
  // Clicks the sidebar link whose visible text matches sectionName.
  // Errors from Playwright locators propagate — they are not caught here.
  // ---------------------------------------------------------------------------
  async navigateTo(sectionName) {
    // The sidebar links have text inside nested <generic> elements.
    // getByRole('link') with name matching works because Playwright computes
    // accessible name from all descendant text content.
    await this.sidebarNav.getByRole('link', { name: new RegExp(sectionName, 'i') }).click();
  }
}

module.exports = { DashboardPage };
