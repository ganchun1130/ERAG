"""Light mint workspace, independent of the operating system's dark preference."""

complete_css = """
:root, .dark, .gradio-container {
    color-scheme: light !important;
    --erag-accent: #37866d;
    --body-background-fill: #fcfdfb;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f2f7f3;
    --background-fill-primary-dark: #ffffff;
    --background-fill-secondary-dark: #f2f7f3;
    --body-text-color: #30483e;
    --body-text-color-subdued: #7c9086;
    --block-background-fill: #ffffff;
    --block-label-background-fill: transparent;
    --block-title-text-color: #6a8175;
    --block-label-text-color: #6a8175;
    --input-background-fill: #ffffff;
    --input-border-color: #e1ebe4;
    --input-placeholder-color: #9cafa3;
    --input-shadow: none;
    --input-shadow-focus: 0 0 0 2px #e4f2e9;
    --border-color-primary: #e6ede7;
    --border-color-accent: #8bc4a8;
    --block-border-color: transparent;
    --block-shadow: none;
    --button-secondary-background-fill: #f2f7f3;
    --button-secondary-background-fill-hover: #e8f2eb;
    --button-secondary-text-color: #567663;
    --button-secondary-border-color: transparent;
    --button-primary-background-fill: #37866d;
    --button-primary-text-color: #ffffff;
    --button-primary-border-color: transparent;
    --link-text-color: #37866d;
    --color-accent: #37866d;
}
gradio-app { background: #fcfdfb !important; }
.gradio-container > .main { padding: 0 !important; }
#workspace-sidebar .form { background: transparent !important; border: 0 !important; box-shadow: none !important; }
body, .dark { background: #fcfdfb !important; color: #30483e !important; }
.gradio-container { max-width: none !important; padding: 0 !important; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif !important; }
footer { display: none !important; }
#workspace-sidebar { background: #f1f7f2 !important; border-right: 1px solid #e9efe8; padding: 0 !important; }
#workspace-sidebar .sidebar-content { padding: 26px 19px 20px !important; scrollbar-width: thin; scrollbar-color: #dce8dd transparent; }
#workspace-sidebar .block { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.brand { display: flex; align-items: center; gap: 12px; margin: 0 3px 28px; }
.brand-mark { display: grid; place-items: center; width: 39px; height: 39px; background: #e0efe3; color: #478a67; border-radius: 14px; font-size: 30px; }
.brand b { display: block; font-size: 20px; letter-spacing: .09em; color: #365745; }
.brand small { display: block; color: #90a28f; font-size: 11px; letter-spacing: .16em; margin-top: 2px; }
#new-chat { background: #e1eee3 !important; color: #437153 !important; border: 1px solid #d7e7d9 !important; box-shadow: none !important; border-radius: 12px; min-height: 44px; font-size: 14px; }
#new-chat:hover { background: #d7e9da !important; }
#history-search input { font-size: 12px; background: #f8fbf7 !important; border: 1px solid #e0e9df !important; border-radius: 10px; padding: 11px 12px; }
#history-list { padding: 8px 0 0 !important; min-height: 130px; max-height: 30vh; overflow-y: auto; }
#history-list > span { padding: 0 8px; color: #9aaa9a; font-size: 11px; letter-spacing: .08em; }
#history-list .wrap { flex-direction: column; gap: 3px; background: transparent !important; }
#history-list label { width: 100%; box-sizing: border-box; border: 0 !important; border-radius: 10px; background: transparent !important; padding: 10px 12px; font-size: 12px; color: #7c8d7e; }
#history-list label:hover { background: #e8f0e7 !important; }
#history-list label:has(input:checked) { background: #e4eee2 !important; color: #52764f !important; }
#history-list input { display: none; }
#history-list label span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sidebar-section { background: transparent !important; border: 0 !important; border-radius: 10px !important; box-shadow: none !important; }
.sidebar-section > button { padding: 10px 8px !important; color: #7a8f7c !important; font-size: 12px; }
#tools-title { margin: 12px 8px 0; }
#tools-title h3 { font-size: 11px; font-weight: 500; color: #9aaa9a; letter-spacing: .05em; }
#workspace-sidebar input, #workspace-sidebar textarea { color: #4a6654 !important; }
#workspace-sidebar .secondary { border-radius: 9px !important; font-size: 12px; }
.sidebar-note { font-size: 10px; line-height: 1.9; color: #a0afa0; padding: 24px 8px 0; }
#chat-main { height: 100dvh; box-sizing: border-box; max-width: 1030px; margin: 0 auto; padding: 0 44px 16px; min-width: 0; gap: 6px; }
.chat-heading { display: flex; align-items: center; justify-content: space-between; min-height: 69px; border-bottom: 1px solid #edf1eb; gap: 20px; }
.chat-heading b { font-size: 13px; font-weight: 500; color: #667e6c; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chat-heading span { font-size: 10px; color: #8ba18d; background: #f0f6ed; border: 1px solid #e9f0e3; padding: 5px 10px; border-radius: 20px; white-space: nowrap; }
#conversation { min-height: 0 !important; flex: 1 1 0 !important; height: auto !important; background: transparent !important; border: 0 !important; box-shadow: none !important; }
#conversation .wrapper, #conversation .wrap, #conversation .bubble-wrap { background: transparent !important; }
#conversation .message { font-size: 14px; line-height: 1.9; color: #3c5143 !important; }
#conversation .user { background: #edf3e9 !important; color: #4e6c4e !important; border: 0 !important; border-radius: 17px 17px 5px 17px !important; }
#conversation .bot { background: transparent !important; border: 0 !important; }
#conversation button { color: #93a390 !important; background: transparent !important; border-color: #ecf1e8 !important; }
.welcome { text-align: center; padding: 5vh 20px 0; }
.welcome-icon { display: grid; place-items: center; width: 72px; height: 72px; border-radius: 25px; background: linear-gradient(135deg,#e9f4e5,#f7f6df); color: #84af76; font-size: 45px; margin: 0 auto 27px; transform: rotate(-8deg); }
.welcome-eyebrow { font-size: 8px; letter-spacing: .22em; color: #a8b49e; margin-bottom: 13px; }
.welcome h1 { font-size: 28px !important; font-weight: 500 !important; color: #526f4d !important; letter-spacing: .04em; }
.welcome p { font-size: 13px !important; line-height: 2 !important; color: #9aa792 !important; margin-top: 18px !important; }
#chat-status { font-size: 11px; color: #87a07b; min-height: 15px; padding-left: 8px; }
#composer { align-items: flex-end; flex-wrap: nowrap; padding: 12px 13px 12px 18px; border: 1px solid #e1eadd; border-radius: 19px; box-shadow: 0 4px 22px #536d3e06; gap: 10px; background: #ffffff; }
#composer:focus-within { border-color: #b9d0ae; box-shadow: 0 0 0 3px #f2f7ed; }
#message-input, #message-input textarea { border: 0 !important; box-shadow: none !important; background: transparent !important; }
#message-input textarea { padding: 4px 0 !important; font-size: 14px; color: #46613e !important; }
#message-input textarea::placeholder { color: #b1bba9; }
#send-message { background: #8caf79 !important; color: white; width: 39px; height: 39px; min-width: 39px !important; border: 0; border-radius: 12px; font-size: 23px; box-shadow: none !important; }
#send-message:hover { background: #769c61 !important; }
#composer-actions { align-items: center; gap: 6px; }
#composer-actions button { border: 0; background: transparent !important; color: #9ba88f; font-size: 10px; box-shadow: none; }
#composer-hint { font-size: 9px; color: #b2bcaa; flex: 1; padding-left: 6px; }
.message details { margin-top: 14px; padding: 8px 12px; border: 1px solid #e5eddf; background: #fafcf7; border-radius: 10px; font-size: 12px; }
.message summary { cursor: pointer; color: #88a278; }
.message details li { margin: 10px 0; line-height: 1.75; }
@media (max-width: 768px) {
    #chat-main { padding: 0 14px 10px; }
    .chat-heading { padding-left: 26px; min-height: 55px; }
    .chat-heading span { display: none; }
    .welcome h1 { font-size: 24px !important; }
    .welcome-eyebrow { font-size: 7px; }
    #composer-hint { font-size: 8px; }
}
"""
