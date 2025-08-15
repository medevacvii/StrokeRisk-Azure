import streamlit as st
# No need for sys or os imports if we're just embedding HTML directly

def model_performance_fairness_page():
    """
    Renders the Model Performance & Fairness page by embedding the complete
    HTML content. It now includes a dynamic header and footer, consistent
    with other pages like Risk Assessment Results.
    """
    # Define the header HTML content (consistent with other pages)
    header_html_content = """
    <header class="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#f1f2f4] px-10 py-3">
        <div class="flex items-center gap-4 text-[#121516]">
            <div class="size-4">
                <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path fill-rule="evenodd" clip-rule="evenodd" d="M24 4H6V17.3333V30.6667H24V44H42V30.6667V17.3333H24V4Z" fill="currentColor"></path>
                </svg>
            </div>
            <h2 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em]">StrokeRisk</h2>
        </div>
        <div class="flex items-center justify-end gap-6 flex-wrap flex-1">
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/01_Home.py">Home</a>
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/07_About_Us.py">About Us</a>
            <a class="text-[#121516] text-sm font-medium leading-normal" href="pages/05_Patient_Records.py">Manage Patient Records</a>
            
            <a href="pages/08_Practitioners_Profile.py" class="flex size-10 items-center justify-center rounded-full overflow-hidden">
                <img src="https://lh3.googleusercontent.com/aida-public/AB6AXuBwkktHQpR_bDtX78dae6gQMFY5NXBfpOazltOY9n2KDjANnnbfMSvj1VWr6h9sdxif9ERNDC4qiRzYq3h5DAPM8-R7pLtxTDUn1yabMKiR-M9aKohcwYezZ-0kO1omt02RpBEHBmpq9JeThPsCa8WcsleL1TSS668g3P_dtzNLUOWlc2lRl_IzpdsrsvTBBl7ypv9s3rjUgLBazzzNUdhIp345xspze6IOTt0Cntw09qi6KJ913o7nc9nSNJdeuqd8gyzf5dsfw1k" alt="Practitioner Profile" class="w-full h-full object-cover">
            </a>

            <a href="pages/13_Logout.py" class="flex min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#3f8abf] text-white text-sm font-bold leading-normal tracking-[0.015em]">
                <span class="truncate">Log Out</span>
            </a>
            </a>
        </div>
    </header>
    """

    # Define the footer HTML content (consistent with pages/03_Risk_Assessment_Results.py)
    footer_html_content = """
    <footer class="flex justify-center">
        <div class="flex flex-1 flex-col">
            <footer class="flex flex-col gap-6 px-5 py-10 text-center @container">
                <div class="flex flex-wrap items-center justify-center gap-6 @[480px]:flex-row @[480px]:justify-around">
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/04_Model_Performance.py">Model Performance</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/10_System_setting.py">System Settings</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/12_Help_and_Support.py">Help & Support</a>
                    <a class="text-[#6a7781] text-base font-normal leading-normal min-w-40" href="pages/13_Logout.py">Log Out</a>
                </div>
                <p class="text-[#6a7781] text-base font-normal leading-normal">© 2025 G4 Pulse. All rights reserved.</p>
            </footer>
        </div>
    </footer>
    """

    full_html_content = f"""
    <html>
    <head>
        <link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin="" />
        <link
            rel="stylesheet"
            as="style"
            onload="this.rel='stylesheet'"
            href="https://fonts.googleapis.com/css2?display=swap&amp;family=Noto+Sans%3Awght%40400%3B500%3B700%3B900&amp;family=Public+Sans%3Awght%40400%3B500%3B700%3B900"
        />

        <title>Model Performance & Fairness</title>
        <link rel="icon" type="image/x-icon" href="data:image/x-icon;base64," />

        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <style>
            /* Ensure the body uses your specified fonts, even within the iframe */
            body {{ font-family: "Public Sans", "Noto Sans", sans-serif; margin: 0; padding: 0; }}
            /* Ensure full height for the layout container within the iframe */
            .relative.flex.size-full.min-h-screen.flex-col {{ min-height: 100vh; }}

            /* Adjustments for width and padding within the embedded HTML */
            .flex.flex-1.justify-center.py-5 {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            .layout-content-container.flex.flex-col.flex-1 {{
                max-width: 100% !important;
                width: 100% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            header.px-10.py-3 {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }}

            .table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-120,
            .table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-240,
            .table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-360 {{
                width: auto !important;
            }}
            table {{
                width: 100%;
                table-layout: fixed;
            }}
            /* Escaped curly braces for @container rules */
            @container(max-width:120px){{.table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-120{{display: none;}}}}
            @container(max-width:240px){{.table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-240{{display: none;}}}}
            @container(max-width:360px){{.table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-360{{display: none;}}}}
        </style>
    </head>
    <body>
        <div class="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden" style='font-family: "Public Sans", "Noto Sans", sans-serif;'>
            <div class="layout-container flex h-full grow flex-col">
                {header_html_content}
                <div class="flex flex-1 justify-center py-5 px-4">
                    <div class="layout-content-container flex flex-col flex-1">
                        <div class="flex flex-wrap justify-between gap-3 p-4">
                            <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight min-w-72">Model Performance &amp; Fairness</p>
                        </div>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Overall Performance</h3>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            StrokeRisk's performance is evaluated using several key metrics to ensure its reliability and effectiveness in predicting stroke risk. These metrics provide insights
                            into the model's accuracy, precision, and ability to distinguish between high-risk and low-risk individuals.
                        </p>
                        <div class="flex flex-wrap gap-4 px-4 py-6">
                            <div class="flex min-w-72 flex-1 flex-col gap-2 rounded-xl border border-[#dde1e3] p-6">
                                <p class="text-[#121516] text-base font-medium leading-normal">AUC-ROC</p>
                                <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight truncate">0.85</p>
                                <p class="text-[#6a7781] text-base font-normal leading-normal">Overall</p>
                                <div class="grid min-h-[180px] grid-flow-col gap-6 grid-rows-[1fr_auto] items-end justify-items-center px-3">
                                    <div class="border-[#6a7781] bg-[#f1f2f4] border-t-2 w-full" style="height: 90%;"></div>
                                    <p class="text-[#6a7781] text-[13px] font-bold leading-normal tracking-[0.015em]">Overall</p>
                                </div>
                            </div>
                            <div class="flex min-w-72 flex-1 flex-col gap-2 rounded-xl border border-[#dde1e3] p-6">
                                <p class="text-[#121516] text-base font-medium leading-normal">F1-Score</p>
                                <p class="text-[#121516] tracking-light text-[32px] font-bold leading-tight truncate">0.78</p>
                                <p class="text-[#6a7781] text-base font-normal leading-normal">Overall</p>
                                <div class="grid min-h-[180px] grid-flow-col gap-6 grid-rows-[1fr_auto] items-end justify-items-center px-3">
                                    <div class="border-[#6a7781] bg-[#f1f2f4] border-t-2 w-full" style="height: 10%;"></div>
                                    <p class="text-[#6a7781] text-[13px] font-bold leading-normal tracking-[0.015em]">Overall</p>
                                </div>
                            </div>
                        </div>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            AUC-ROC (Area Under the Receiver Operating Characteristic curve) measures the model's ability to distinguish between patients who will experience a stroke and those
                            who will not. A score of 0.85 indicates excellent discriminatory power. The F1-Score, at 0.78, balances precision and recall, providing a comprehensive measure of the
                            model's accuracy.
                        </p>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Fairness Audit</h3>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            To ensure fairness and mitigate potential biases, StrokeRisk undergoes rigorous fairness audits. These audits assess the model's performance across different
                            demographic subgroups, including gender, age, and race/ethnicity. The results are disaggregated to identify any disparities and ensure equitable predictions for all
                            patients.
                        </p>
                        <div class="px-4 py-3 @container">
                            <div class="flex overflow-hidden rounded-xl border border-[#dde1e3] bg-white">
                                <table class="flex-1">
                                    <thead>
                                        <tr class="bg-white">
                                            <th class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-120 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">
                                                Demographic Group
                                            </th>
                                            <th class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-240 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">AUC-ROC</th>
                                            <th class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-360 px-4 py-3 text-left text-[#121516] w-[400px] text-sm font-medium leading-normal">
                                                F1-Score
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-120 h-[72px] px-4 py-2 w-[400px] text-[#121516] text-sm font-normal leading-normal">Gender</td>
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                0.84 (Male), 0.86 (Female)
                                            </td>
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-360 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                0.77 (Male), 0.79 (Female)
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-120 h-[72px] px-4 py-2 w-[400px] text-[#121516] text-sm font-normal leading-normal">Age</td>
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                0.83 (18-45), 0.85 (46-65), 0.87 (65+)
                                            </td>
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-360 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                0.76 (18-45), 0.78 (46-65), 0.80 (65+)
                                            </td>
                                        </tr>
                                        <tr class="border-t border-t-[#dde1e3]">
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-120 h-[72px] px-4 py-2 w-[400px] text-[#121516] text-sm font-normal leading-normal">
                                                Race/Ethnicity
                                            </td>
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-240 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                0.84 (Group A), 0.85 (Group B), 0.86 (Group C)
                                            </td>
                                            <td class="table-0d16ee16-aabc-4e63-a0bd-f9be3b680310-column-360 h-[72px] px-4 py-2 w-[400px] text-[#6a7781] text-sm font-normal leading-normal">
                                                0.77 (Group A), 0.78 (Group B), 0.79 (Group C)
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            The table above presents disaggregated performance metrics. While some minor variations exist, the model demonstrates consistent performance across subgroups. Any
                            significant disparities are addressed through ongoing model refinement and data rebalancing.
                        </p>
                        <h3 class="text-[#121516] text-lg font-bold leading-tight tracking-[-0.015em] px-4 pb-2 pt-4">Data and Limitations</h3>
                        <p class="text-[#121516] text-base font-normal leading-normal pb-3 pt-1 px-4">
                            StrokeRisk is trained on a diverse dataset encompassing demographic, medical, and lifestyle factors. However, like all predictive models, it has limitations. The
                            model's accuracy is contingent on the quality and representativeness of the training data. We continuously update the model with new data and refine its algorithms to
                            improve performance and address any identified limitations. Future work includes incorporating additional data sources and exploring advanced modeling techniques to
                            enhance predictive capabilities.
                        </p>
                    </div>
                </div>
                {footer_html_content}
            </div>
        </div>
    </body>
    </html>
    """
    # Use st.components.v1.html to embed the full HTML content
    st.components.v1.html(full_html_content, height=2000, scrolling=False) # Increased height for this page

# Call the function to render the page.
model_performance_fairness_page()
