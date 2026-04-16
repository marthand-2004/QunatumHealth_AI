# Requirements Document

## Introduction

This feature covers the automated test suite for the Sign-Up and Sign-In modules of the Owlet Campus web application (https://owlet-campus.com/). The suite is built with Playwright and JavaScript, following the Page Object Model (POM) pattern. It converts existing manual test cases into maintainable, production-ready automated scripts covering both positive and negative scenarios.

## Glossary

- **Test_Suite**: The complete collection of Playwright test files targeting the Owlet Campus application.
- **SignUp_Page**: The Page Object representing the registration form with Name, Email, Password, Confirm Password fields and a Register button.
- **SignIn_Page**: The Page Object representing the login form with Email and Password fields and a Login button.
- **POM**: Page Object Model — a design pattern that encapsulates page selectors and interactions into reusable class objects.
- **Locator**: A Playwright `page.locator()` or `getByRole()` reference used to identify a UI element.
- **Assertion**: A Playwright `expect()` call that verifies the state of the application.
- **BeforeEach_Hook**: A `test.beforeEach()` block that runs setup logic (e.g., navigation) before every test case.
- **Valid_Credentials**: An email/password pair that exists in the application and is accepted by the Sign-In flow.
- **Invalid_Credentials**: An email/password pair that does not match any registered account.

---

## Requirements

### Requirement 1: Project Structure and Configuration

**User Story:** As a QA engineer, I want a well-structured Playwright project, so that the test suite is maintainable and easy to extend.

#### Acceptance Criteria

1. THE Test_Suite SHALL be organised using the Page Object Model pattern, with page objects stored separately from test specs.
2. THE Test_Suite SHALL use `async/await` syntax throughout all test files and page objects.
3. THE Test_Suite SHALL include a `playwright.config.js` file that sets the base URL to `https://owlet-campus.com/`.
4. THE Test_Suite SHALL use `page.locator()` or `getByRole()` for all element selection.
5. THE Test_Suite SHALL include inline comments explaining the purpose of each logical block.

---

### Requirement 2: Sign-Up Page Object

**User Story:** As a QA engineer, I want a reusable Sign-Up page object, so that registration interactions are not duplicated across test cases.

#### Acceptance Criteria

1. THE SignUp_Page SHALL expose a `navigate()` method that navigates to the registration page.
2. THE SignUp_Page SHALL expose a `fillForm(name, email, password, confirmPassword)` method that fills all four registration fields.
3. THE SignUp_Page SHALL expose a `submit()` method that clicks the Register button.
4. THE SignUp_Page SHALL expose a `register(name, email, password, confirmPassword)` convenience method that calls `fillForm` then `submit`.
5. THE SignUp_Page SHALL expose locator accessors for the Name, Email, Password, Confirm Password fields and the Register button so that tests can assert field states directly.
6. IF a selector cannot be resolved at runtime, THEN THE SignUp_Page SHALL surface the Playwright locator error without swallowing it.

---

### Requirement 3: Sign-In Page Object

**User Story:** As a QA engineer, I want a reusable Sign-In page object, so that login interactions are not duplicated across test cases.

#### Acceptance Criteria

1. THE SignIn_Page SHALL expose a `navigate()` method that navigates to the login page.
2. THE SignIn_Page SHALL expose a `fillForm(email, password)` method that fills the Email and Password fields.
3. THE SignIn_Page SHALL expose a `submit()` method that clicks the Login button.
4. THE SignIn_Page SHALL expose a `login(email, password)` convenience method that calls `fillForm` then `submit`.
5. THE SignIn_Page SHALL expose locator accessors for the Email and Password fields and the Login button so that tests can assert field states directly.
6. IF a selector cannot be resolved at runtime, THEN THE SignIn_Page SHALL surface the Playwright locator error without swallowing it.

---

### Requirement 4: Sign-Up Test Scenarios

**User Story:** As a QA engineer, I want automated Sign-Up tests, so that registration behaviour is continuously verified.

#### Acceptance Criteria

1. WHEN valid Name, Email, Password, and matching Confirm Password are submitted, THE Test_Suite SHALL assert that the application navigates away from the registration page or displays a success indicator.
2. WHEN an already-registered email is submitted, THE Test_Suite SHALL assert that an error message indicating a duplicate account is visible.
3. WHEN an email value that does not conform to standard email format is submitted, THE Test_Suite SHALL assert that a validation error is visible before or after form submission.
4. WHEN the Password and Confirm Password fields contain different values, THE Test_Suite SHALL assert that a mismatch error message is visible.
5. WHEN the registration form is submitted with one or more empty required fields, THE Test_Suite SHALL assert that a required-field validation message is visible for each empty field.
6. THE Test_Suite SHALL use a `test.beforeEach()` hook in the Sign-Up spec to navigate to the registration page before each test case.

---

### Requirement 5: Sign-In Test Scenarios

**User Story:** As a QA engineer, I want automated Sign-In tests, so that login behaviour is continuously verified.

#### Acceptance Criteria

1. WHEN Valid_Credentials are submitted, THE Test_Suite SHALL assert that the application navigates to the authenticated home or dashboard page.
2. WHEN Invalid_Credentials are submitted, THE Test_Suite SHALL assert that an error message indicating invalid credentials is visible.
3. WHEN the login form is submitted with both Email and Password fields empty, THE Test_Suite SHALL assert that a required-field or validation message is visible.
4. WHEN a registered email is submitted with an incorrect password, THE Test_Suite SHALL assert that an authentication error message is visible.
5. THE Test_Suite SHALL use a `test.beforeEach()` hook in the Sign-In spec to navigate to the login page before each test case.

---

### Requirement 6: Assertions and Reporting

**User Story:** As a QA engineer, I want meaningful assertions and clear test descriptions, so that failures are easy to diagnose.

#### Acceptance Criteria

1. THE Test_Suite SHALL use `expect()` from `@playwright/test` for all assertions.
2. WHEN a test fails, THE Test_Suite SHALL produce a failure message that identifies the specific assertion that did not pass.
3. THE Test_Suite SHALL use descriptive strings in `test()` calls that clearly state the scenario being verified (e.g., `'should show error for mismatched passwords'`).
4. WHERE Playwright's built-in HTML reporter is available, THE Test_Suite SHALL be configured to generate an HTML report on test completion.
