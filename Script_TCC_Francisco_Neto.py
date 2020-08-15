




<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://github.githubassets.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" media="all" integrity="sha512-jFUBCdWOA1Ov3xo3oFMBwsdP4Up2K1bRnP4QYI5WqvpaIYxWVek89k2M0oyTbNhYMViGtxJB3Vdwcw8ln8hGQw==" rel="stylesheet" href="https://github.githubassets.com/assets/frameworks-8c550109d58e0353afdf1a37a05301c2.css" />
  
    <link crossorigin="anonymous" media="all" integrity="sha512-W5+OY/xW8V+LIAgg/LOCwIIp6enwXwXxqbFsvPpDC4MJCzZTZ1Wln8eDnNTHs4+xpP+F7/47pQAmpcLATRI9FQ==" rel="stylesheet" href="https://github.githubassets.com/assets/github-5b9f8e63fc56f15f8b200820fcb382c0.css" />
    
    
    
    


  <meta name="viewport" content="width=device-width">
  
  <title>PUC_Minas_TCC/Script_TCC_Francisco_Neto.py at master · itfrancisconeto/PUC_Minas_TCC</title>
    <meta name="description" content="Contribute to itfrancisconeto/PUC_Minas_TCC development by creating an account on GitHub.">
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">
  <meta name="apple-itunes-app" content="app-id=1477376905">

    <meta name="twitter:image:src" content="https://avatars0.githubusercontent.com/u/40370984?s=400&amp;v=4" /><meta name="twitter:site" content="@github" /><meta name="twitter:card" content="summary" /><meta name="twitter:title" content="itfrancisconeto/PUC_Minas_TCC" /><meta name="twitter:description" content="Contribute to itfrancisconeto/PUC_Minas_TCC development by creating an account on GitHub." />
    <meta property="og:image" content="https://avatars0.githubusercontent.com/u/40370984?s=400&amp;v=4" /><meta property="og:site_name" content="GitHub" /><meta property="og:type" content="object" /><meta property="og:title" content="itfrancisconeto/PUC_Minas_TCC" /><meta property="og:url" content="https://github.com/itfrancisconeto/PUC_Minas_TCC" /><meta property="og:description" content="Contribute to itfrancisconeto/PUC_Minas_TCC development by creating an account on GitHub." />

  <link rel="assets" href="https://github.githubassets.com/">
    <link rel="shared-web-socket" href="wss://alive.github.com/_sockets/u/40370984/ws?session=eyJ2IjoiVjMiLCJ1Ijo0MDM3MDk4NCwicyI6NTQ5MTAyMDQ2LCJjIjoxMzg3Njk1MDkzLCJ0IjoxNTk3NTAyMjkxfQ==--437b3c86b11b124c1497f8b7ba7fe69cda1648ba1a2f4cffa66282504ec9739d" data-refresh-url="/_ws">
  <link rel="sudo-modal" href="/sessions/sudo_modal">

  <meta name="request-id" content="D675:4A2F:2CB815A:429BA2D:5F37F33D" data-pjax-transient="true" /><meta name="html-safe-nonce" content="39fa7ed48337a3e8e27ab97c9aa577e685163512" data-pjax-transient="true" /><meta name="visitor-payload" content="eyJyZWZlcnJlciI6Imh0dHBzOi8vZ2l0aHViLmNvbS9pdGZyYW5jaXNjb25ldG8vUFVDX01pbmFzX1RDQyIsInJlcXVlc3RfaWQiOiJENjc1OjRBMkY6MkNCODE1QTo0MjlCQTJEOjVGMzdGMzNEIiwidmlzaXRvcl9pZCI6Ijc2MjAwNzIyNzg4MjUzMjUzNjQiLCJyZWdpb25fZWRnZSI6ImlhZCIsInJlZ2lvbl9yZW5kZXIiOiJpYWQifQ==" data-pjax-transient="true" /><meta name="visitor-hmac" content="9ba13e40489f2a37d128b745b955a5138ab59ffccf7e7616dde95498f7d33f35" data-pjax-transient="true" />

    <meta name="hovercard-subject-tag" content="repository:264758279" data-pjax-transient>


  <meta name="github-keyboard-shortcuts" content="repository,source-code" data-pjax-transient="true" />

  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="c1kuD-K2HIVF635lypcsWPoD4kilo5-jA_wBFyT4uMY">
  <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
  <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
  <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">

  <meta name="octolytics-host" content="collector.githubapp.com" /><meta name="octolytics-app-id" content="github" /><meta name="octolytics-event-url" content="https://collector.githubapp.com/github-external/browser_event" /><meta name="octolytics-dimension-ga_id" content="" class="js-octo-ga-id" /><meta name="octolytics-actor-id" content="40370984" /><meta name="octolytics-actor-login" content="itfrancisconeto" /><meta name="octolytics-actor-hash" content="cae11d3fc7b177a54e4dee5113b1c5740d8a587399c41cf82377ba157a8bbcee" />

  <meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" />

  




    <meta name="google-analytics" content="UA-3769691-2">

  <meta class="js-ga-set" name="userId" content="d294443a4f1d5e265c78d5f68e88c9dc">

<meta class="js-ga-set" name="dimension10" content="Responsive" data-pjax-transient>

<meta class="js-ga-set" name="dimension1" content="Logged In">



  

      <meta name="hostname" content="github.com">
    <meta name="user-login" content="itfrancisconeto">


      <meta name="expected-hostname" content="github.com">

      <meta name="js-proxy-site-detection-payload" content="ZDRiYmVkNzI5OTJkMGU2MGU4ZTlmNzhjZDBiZTY1MmQ5M2QzNDllNGFiOGE4M2UxMDgyMGU2NDdmMmZhYmY0ZXx7InJlbW90ZV9hZGRyZXNzIjoiMTg5LjkwLjEyMy4xODYiLCJyZXF1ZXN0X2lkIjoiRDY3NTo0QTJGOjJDQjgxNUE6NDI5QkEyRDo1RjM3RjMzRCIsInRpbWVzdGFtcCI6MTU5NzUwMjI5MSwiaG9zdCI6ImdpdGh1Yi5jb20ifQ==">

    <meta name="enabled-features" content="MARKETPLACE_PENDING_INSTALLATIONS">

  <meta http-equiv="x-pjax-version" content="66d8b273ec5f3a08783c361b2e3f6945">
  

      <link href="https://github.com/itfrancisconeto/PUC_Minas_TCC/commits/master.atom" rel="alternate" title="Recent Commits to PUC_Minas_TCC:master" type="application/atom+xml">

  <meta name="go-import" content="github.com/itfrancisconeto/PUC_Minas_TCC git https://github.com/itfrancisconeto/PUC_Minas_TCC.git">

  <meta name="octolytics-dimension-user_id" content="40370984" /><meta name="octolytics-dimension-user_login" content="itfrancisconeto" /><meta name="octolytics-dimension-repository_id" content="264758279" /><meta name="octolytics-dimension-repository_nwo" content="itfrancisconeto/PUC_Minas_TCC" /><meta name="octolytics-dimension-repository_public" content="true" /><meta name="octolytics-dimension-repository_is_fork" content="false" /><meta name="octolytics-dimension-repository_network_root_id" content="264758279" /><meta name="octolytics-dimension-repository_network_root_nwo" content="itfrancisconeto/PUC_Minas_TCC" /><meta name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" content="true" />


    <link rel="canonical" href="https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://github.githubassets.com/pinned-octocat.svg" color="#000000">
  <link rel="alternate icon" class="js-site-favicon" type="image/png" href="https://github.githubassets.com/favicons/favicon.png">
  <link rel="icon" class="js-site-favicon" type="image/svg+xml" href="https://github.githubassets.com/favicons/favicon.svg">

<meta name="theme-color" content="#1e2327">


  <link rel="manifest" href="/manifest.json" crossOrigin="use-credentials">

  </head>

  <body class="logged-in env-production page-responsive page-blob">
    

    <input type="hidden" value="i1w5NfAuefHTLvxU2WwRo22kDylWRmU/WeQ8dfXib+5FvGJGqoDcTP3s2E1rq76rm8ELtjPs8zSI6SJRq48wgw==" data-csrf="true" class="js-data-websocket-refresh-csrf" />

    <div class="position-relative js-header-wrapper ">
      <a href="#start-of-content" class="p-3 bg-blue text-white show-on-focus js-skip-to-content">Skip to content</a>
      <span class="progress-pjax-loader width-full js-pjax-loader-bar Progress position-fixed">
    <span style="background-color: #79b8ff;width: 0%;" class="Progress-item progress-pjax-loader-bar "></span>
</span>      
      



          <header class="Header py-md-0 js-details-container Details flex-wrap flex-md-nowrap px-3" role="banner">
  <div class="Header-item d-none d-md-flex">
    <a class="Header-link" href="https://github.com/" data-hotkey="g d"
  aria-label="Homepage " data-ga-click="Header, go to dashboard, icon:logo">
  <svg class="octicon octicon-mark-github v-align-middle" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>

  </div>

  <div class="Header-item d-md-none">
    <button class="Header-link btn-link js-details-target" type="button" aria-label="Toggle navigation" aria-expanded="false">
      <svg height="24" class="octicon octicon-three-bars" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M1 2.75A.75.75 0 011.75 2h12.5a.75.75 0 110 1.5H1.75A.75.75 0 011 2.75zm0 5A.75.75 0 011.75 7h12.5a.75.75 0 110 1.5H1.75A.75.75 0 011 7.75zM1.75 12a.75.75 0 100 1.5h12.5a.75.75 0 100-1.5H1.75z"></path></svg>
    </button>
  </div>

  <div class="Header-item Header-item--full flex-column flex-md-row width-full flex-order-2 flex-md-order-none mr-0 mr-md-3 mt-3 mt-md-0 Details-content--hidden-not-important d-md-flex">
        <div class="header-search header-search-current js-header-search-current flex-auto  flex-self-stretch flex-md-self-auto mr-0 mr-md-3 mb-3 mb-md-0 scoped-search site-scoped-search js-site-search position-relative js-jump-to js-header-search-current-jump-to"
  role="combobox"
  aria-owns="jump-to-results"
  aria-label="Search or jump to"
  aria-haspopup="listbox"
  aria-expanded="false"
>
  <div class="position-relative">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-site-search-form" role="search" aria-label="Site" data-scope-type="Repository" data-scope-id="264758279" data-scoped-search-url="/itfrancisconeto/PUC_Minas_TCC/search" data-unscoped-search-url="/search" action="/itfrancisconeto/PUC_Minas_TCC/search" accept-charset="UTF-8" method="get">
      <label class="form-control input-sm header-search-wrapper p-0 header-search-wrapper-jump-to position-relative d-flex flex-justify-between flex-items-center js-chromeless-input-container">
        <input type="text"
          class="form-control input-sm header-search-input jump-to-field js-jump-to-field js-site-search-focus js-site-search-field is-clearable"
          data-hotkey="s,/"
          name="q"
          value=""
          placeholder="Search or jump to…"
          data-unscoped-placeholder="Search or jump to…"
          data-scoped-placeholder="Search or jump to…"
          autocapitalize="off"
          aria-autocomplete="list"
          aria-controls="jump-to-results"
          aria-label="Search or jump to…"
          data-jump-to-suggestions-path="/_graphql/GetSuggestedNavigationDestinations"
          spellcheck="false"
          autocomplete="off"
          >
          <input type="hidden" value="adyUIkbpb20X3PZbBOqPKzWH9zapAB3V+/Rbka01k8qrXIgoh+WGtpiKVC0qxfT5K9fWMzNz/5oH+p/pCIWdNw==" data-csrf="true" class="js-data-jump-to-suggestions-path-csrf" />
          <input type="hidden" class="js-site-search-type-field" name="type" >
            <img src="https://github.githubassets.com/images/search-key-slash.svg" alt="" class="mr-2 header-search-key-slash">

            <div class="Box position-absolute overflow-hidden d-none jump-to-suggestions js-jump-to-suggestions-container">
              
<ul class="d-none js-jump-to-suggestions-template-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-suggestion" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

</ul>

<ul class="d-none js-jump-to-no-results-template-container">
  <li class="d-flex flex-justify-center flex-items-center f5 d-none js-jump-to-suggestion p-2">
    <span class="text-gray">No suggested jump to results</span>
  </li>
</ul>

<ul id="jump-to-results" role="listbox" class="p-0 m-0 js-navigation-container jump-to-suggestions-results-container js-jump-to-suggestions-results-container">
  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-scoped-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>

  

<li class="d-flex flex-justify-start flex-items-center p-0 f5 navigation-item js-navigation-item js-jump-to-global-search d-none" role="option">
  <a tabindex="-1" class="no-underline d-flex flex-auto flex-items-center jump-to-suggestions-path js-jump-to-suggestion-path js-navigation-open p-2" href="">
    <div class="jump-to-octicon js-jump-to-octicon flex-shrink-0 mr-2 text-center d-none">
      <svg height="16" width="16" class="octicon octicon-repo flex-shrink-0 js-jump-to-octicon-repo d-none" title="Repository" aria-label="Repository" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-project flex-shrink-0 js-jump-to-octicon-project d-none" title="Project" aria-label="Project" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
      <svg height="16" width="16" class="octicon octicon-search flex-shrink-0 js-jump-to-octicon-search d-none" title="Search" aria-label="Search" viewBox="0 0 16 16" version="1.1" role="img"><path fill-rule="evenodd" d="M11.5 7a4.499 4.499 0 11-8.998 0A4.499 4.499 0 0111.5 7zm-.82 4.74a6 6 0 111.06-1.06l3.04 3.04a.75.75 0 11-1.06 1.06l-3.04-3.04z"></path></svg>
    </div>

    <img class="avatar mr-2 flex-shrink-0 js-jump-to-suggestion-avatar d-none" alt="" aria-label="Team" src="" width="28" height="28">

    <div class="jump-to-suggestion-name js-jump-to-suggestion-name flex-auto overflow-hidden text-left no-wrap css-truncate css-truncate-target">
    </div>

    <div class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none js-jump-to-badge-search">
      <span class="js-jump-to-badge-search-text-default d-none" aria-label="in this repository">
        In this repository
      </span>
      <span class="js-jump-to-badge-search-text-global d-none" aria-label="in all of GitHub">
        All GitHub
      </span>
      <span aria-hidden="true" class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>

    <div aria-hidden="true" class="border rounded-1 flex-shrink-0 bg-gray px-1 text-gray-light ml-1 f6 d-none d-on-nav-focus js-jump-to-badge-jump">
      Jump to
      <span class="d-inline-block ml-1 v-align-middle">↵</span>
    </div>
  </a>
</li>


    <li class="d-flex flex-justify-center flex-items-center p-0 f5 js-jump-to-suggestion">
      <img src="https://github.githubassets.com/images/spinners/octocat-spinner-128.gif" alt="Octocat Spinner Icon" class="m-2" width="28">
    </li>
</ul>

            </div>
      </label>
</form>  </div>
</div>


    <nav class="d-flex flex-column flex-md-row flex-self-stretch flex-md-self-auto" aria-label="Global">
    <a class="Header-link py-md-3 d-block d-md-none py-2 border-top border-md-top-0 border-white-fade-15" data-ga-click="Header, click, Nav menu - item:dashboard:user" aria-label="Dashboard" href="/dashboard">
      Dashboard
</a>
  <a class="js-selected-navigation-item Header-link py-md-3  mr-0 mr-md-3 py-2 border-top border-md-top-0 border-white-fade-15" data-hotkey="g p" data-ga-click="Header, click, Nav menu - item:pulls context:user" aria-label="Pull requests you created" data-selected-links="/pulls /pulls/assigned /pulls/mentioned /pulls" href="/pulls">
      Pull<span class="d-inline d-md-none d-lg-inline"> request</span>s
</a>
  <a class="js-selected-navigation-item Header-link py-md-3  mr-0 mr-md-3 py-2 border-top border-md-top-0 border-white-fade-15" data-hotkey="g i" data-ga-click="Header, click, Nav menu - item:issues context:user" aria-label="Issues you created" data-selected-links="/issues /issues/assigned /issues/mentioned /issues" href="/issues">
    Issues
</a>

    <div class="mr-0 mr-md-3 py-2 py-md-0 border-top border-md-top-0 border-white-fade-15">
      <a class="js-selected-navigation-item Header-link py-md-3 d-inline-block" data-ga-click="Header, click, Nav menu - item:marketplace context:user" data-octo-click="marketplace_click" data-octo-dimensions="location:nav_bar" data-selected-links=" /marketplace" href="/marketplace">
        Marketplace
</a>      

    </div>

  <a class="js-selected-navigation-item Header-link py-md-3  mr-0 mr-md-3 py-2 border-top border-md-top-0 border-white-fade-15" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship showcases showcases_search showcases_landing /explore" href="/explore">
    Explore
</a>


    <a class="Header-link d-block d-md-none mr-0 mr-md-3 py-2 py-md-3 border-top border-md-top-0 border-white-fade-15" href="/itfrancisconeto">
      <img class="avatar avatar-user" src="https://avatars1.githubusercontent.com/u/40370984?s=40&amp;v=4" width="20" height="20" alt="@itfrancisconeto" />
      itfrancisconeto
</a>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form action="/logout" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="nBj/wloU6k6ivPVMXEv04m5rxq8cIfNbDTzKQujXJfiN5xqDyMRKOaQYO1v2LX0JxJcrUAPuqYFfgpVUKzxxrg==" />
      <button type="submit" class="Header-link mr-0 mr-md-3 py-2 py-md-3 border-top border-md-top-0 border-white-fade-15 d-md-none btn-link d-block width-full text-left" data-ga-click="Header, sign out, icon:logout" style="padding-left: 2px;">
        <svg class="octicon octicon-sign-out v-align-middle" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M2 2.75C2 1.784 2.784 1 3.75 1h2.5a.75.75 0 010 1.5h-2.5a.25.25 0 00-.25.25v10.5c0 .138.112.25.25.25h2.5a.75.75 0 010 1.5h-2.5A1.75 1.75 0 012 13.25V2.75zm10.44 4.5H6.75a.75.75 0 000 1.5h5.69l-1.97 1.97a.75.75 0 101.06 1.06l3.25-3.25a.75.75 0 000-1.06l-3.25-3.25a.75.75 0 10-1.06 1.06l1.97 1.97z"></path></svg>
        Sign out
      </button>
</form></nav>

  </div>

  <div class="Header-item Header-item--full flex-justify-center d-md-none position-relative">
    <a class="Header-link" href="https://github.com/" data-hotkey="g d"
  aria-label="Homepage " data-ga-click="Header, go to dashboard, icon:logo">
  <svg class="octicon octicon-mark-github v-align-middle" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>

  </div>

  <div class="Header-item mr-0 mr-md-3 flex-order-1 flex-md-order-none">
    

    <notification-indicator class="js-socket-channel" data-channel="eyJjIjoibm90aWZpY2F0aW9uLWNoYW5nZWQ6NDAzNzA5ODQiLCJ0IjoxNTk3NTAyMjkxfQ==--f24fadb89a1fed725a13a06db354724d4149f6245bfbdbc046d39a4e34e2503b">
      <a href="/notifications"
         class="Header-link notification-indicator position-relative tooltipped tooltipped-sw"
         aria-label="You have no unread notifications"
         data-hotkey="g n"
         data-ga-click="Header, go to notifications, icon:read"
         data-target="notification-indicator.link">
         <span class="mail-status " data-target="notification-indicator.modifier"></span>
         <svg class="octicon octicon-bell" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M8 16a2 2 0 001.985-1.75c.017-.137-.097-.25-.235-.25h-3.5c-.138 0-.252.113-.235.25A2 2 0 008 16z"></path><path fill-rule="evenodd" d="M8 1.5A3.5 3.5 0 004.5 5v2.947c0 .346-.102.683-.294.97l-1.703 2.556a.018.018 0 00-.003.01l.001.006c0 .002.002.004.004.006a.017.017 0 00.006.004l.007.001h10.964l.007-.001a.016.016 0 00.006-.004.016.016 0 00.004-.006l.001-.007a.017.017 0 00-.003-.01l-1.703-2.554a1.75 1.75 0 01-.294-.97V5A3.5 3.5 0 008 1.5zM3 5a5 5 0 0110 0v2.947c0 .05.015.098.042.139l1.703 2.555A1.518 1.518 0 0113.482 13H2.518a1.518 1.518 0 01-1.263-2.36l1.703-2.554A.25.25 0 003 7.947V5z"></path></svg>
      </a>
    </notification-indicator>

  </div>


  <div class="Header-item position-relative d-none d-md-flex">
    <details class="details-overlay details-reset">
  <summary class="Header-link"
      aria-label="Create new…"
      data-ga-click="Header, create new, icon:add">
    <svg class="octicon octicon-plus" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 2a.75.75 0 01.75.75v4.5h4.5a.75.75 0 010 1.5h-4.5v4.5a.75.75 0 01-1.5 0v-4.5h-4.5a.75.75 0 010-1.5h4.5v-4.5A.75.75 0 018 2z"></path></svg> <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="dropdown-menu dropdown-menu-sw mt-n2">
    
<a role="menuitem" class="dropdown-item" href="/new" data-ga-click="Header, create new repository">
  New repository
</a>

  <a role="menuitem" class="dropdown-item" href="/new/import" data-ga-click="Header, import a repository">
    Import repository
  </a>

<a role="menuitem" class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, create new gist">
  New gist
</a>

  <a role="menuitem" class="dropdown-item" href="/organizations/new" data-ga-click="Header, create new organization">
    New organization
  </a>


  <div role="none" class="dropdown-divider"></div>
  <div class="dropdown-header">
    <span title="itfrancisconeto/PUC_Minas_TCC">This repository</span>
  </div>
    <a role="menuitem" class="dropdown-item" href="/itfrancisconeto/PUC_Minas_TCC/issues/new/choose" data-ga-click="Header, create new issue" data-skip-pjax>
      New issue
    </a>


  </details-menu>
</details>

  </div>

  <div class="Header-item position-relative mr-0 d-none d-md-flex">
    
  <details class="details-overlay details-reset js-feature-preview-indicator-container" data-feature-preview-indicator-src="/users/itfrancisconeto/feature_preview/indicator_check">

  <summary class="Header-link"
    aria-label="View profile and more"
    data-ga-click="Header, show menu, icon:avatar">
    <img
  alt="@itfrancisconeto"
  width="20"
  height="20"
  src="https://avatars2.githubusercontent.com/u/40370984?s=60&amp;v=4"
  class="avatar avatar-user " />

      <span class="feature-preview-indicator js-feature-preview-indicator" style="top: 10px;" hidden></span>
    <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="dropdown-menu dropdown-menu-sw mt-n2" style="width: 180px" >
    <div class="header-nav-current-user css-truncate"><a role="menuitem" class="no-underline user-profile-link px-3 pt-2 pb-2 mb-n2 mt-n1 d-block" href="/itfrancisconeto" data-ga-click="Header, go to profile, text:Signed in as">Signed in as <strong class="css-truncate-target">itfrancisconeto</strong></a></div>
    <div role="none" class="dropdown-divider"></div>

      <div class="pl-3 pr-3 f6 user-status-container js-user-status-context lh-condensed" data-url="/users/status?compact=1&amp;link_mentions=0&amp;truncate=1">
        
<div class="js-user-status-container rounded-1 px-2 py-1 mt-2 border"
  data-team-hovercards-enabled>
  <details class="js-user-status-details details-reset details-overlay details-overlay-dark">
    <summary class="btn-link btn-block link-gray no-underline js-toggle-user-status-edit toggle-user-status-edit "
      role="menuitem" data-hydro-click="{&quot;event_type&quot;:&quot;user_profile.click&quot;,&quot;payload&quot;:{&quot;profile_user_id&quot;:40370984,&quot;target&quot;:&quot;EDIT_USER_STATUS&quot;,&quot;user_id&quot;:40370984,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;}}" data-hydro-click-hmac="31c1bff2eb0c41c9e33ed5da05f7c1cfdbd4343d79c34fa8bb886b9b9323903f">
      <div class="d-flex flex-items-center flex-items-stretch">
        <div class="f6 lh-condensed user-status-header d-flex user-status-emoji-only-header circle">
          <div class="user-status-emoji-container flex-shrink-0 mr-2 d-flex flex-items-center flex-justify-center lh-condensed-ultra v-align-bottom">
            <svg class="octicon octicon-smiley" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM5 8a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zM5.32 9.636a.75.75 0 011.038.175l.007.009c.103.118.22.222.35.31.264.178.683.37 1.285.37.602 0 1.02-.192 1.285-.371.13-.088.247-.192.35-.31l.007-.008a.75.75 0 111.222.87l-.614-.431c.614.43.614.431.613.431v.001l-.001.002-.002.003-.005.007-.014.019a1.984 1.984 0 01-.184.213c-.16.166-.338.316-.53.445-.63.418-1.37.638-2.127.629-.946 0-1.652-.308-2.126-.63a3.32 3.32 0 01-.715-.657l-.014-.02-.005-.006-.002-.003v-.002h-.001l.613-.432-.614.43a.75.75 0 01.183-1.044h.001z"></path></svg>
          </div>
        </div>
        <div class="
          
           user-status-message-wrapper f6 min-width-0"
           style="line-height: 20px;" >
          <div class="css-truncate css-truncate-target width-fit text-gray-dark text-left">
              <span class="text-gray">Set status</span>
          </div>
        </div>
      </div>
    </summary>
    <details-dialog class="details-dialog rounded-1 anim-fade-in fast Box Box--overlay" role="dialog" tabindex="-1">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="position-relative flex-auto js-user-status-form" action="/users/status?circle=0&amp;compact=1&amp;link_mentions=0&amp;truncate=1" accept-charset="UTF-8" method="post"><input type="hidden" name="_method" value="put" /><input type="hidden" name="authenticity_token" value="vKZIIlqYOj/aQ2lP0UrMbl0MwDfbUFk0ZcgUxbc6fjFRtRP+qyGqaiK8ZJGfmhqyxzwdI0Vpy/yXGZX71IFkZw==" />
        <div class="Box-header bg-gray border-bottom p-3">
          <button class="Box-btn-octicon js-toggle-user-status-edit btn-octicon float-right" type="reset" aria-label="Close dialog" data-close-dialog>
            <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
          </button>
          <h3 class="Box-title f5 text-bold text-gray-dark">Edit status</h3>
        </div>
        <input type="hidden" name="emoji" class="js-user-status-emoji-field" value="">
        <input type="hidden" name="organization_id" class="js-user-status-org-id-field" value="">
        <div class="px-3 py-2 text-gray-dark">
          <div class="js-characters-remaining-container position-relative mt-2">
            <div class="input-group d-table form-group my-0 js-user-status-form-group">
              <span class="input-group-button d-table-cell v-align-middle" style="width: 1%">
                <button type="button" aria-label="Choose an emoji" class="btn-outline btn js-toggle-user-status-emoji-picker btn-open-emoji-picker p-0">
                  <span class="js-user-status-original-emoji" hidden></span>
                  <span class="js-user-status-custom-emoji"></span>
                  <span class="js-user-status-no-emoji-icon" >
                    <svg class="octicon octicon-smiley" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM5 8a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zM5.32 9.636a.75.75 0 011.038.175l.007.009c.103.118.22.222.35.31.264.178.683.37 1.285.37.602 0 1.02-.192 1.285-.371.13-.088.247-.192.35-.31l.007-.008a.75.75 0 111.222.87l-.614-.431c.614.43.614.431.613.431v.001l-.001.002-.002.003-.005.007-.014.019a1.984 1.984 0 01-.184.213c-.16.166-.338.316-.53.445-.63.418-1.37.638-2.127.629-.946 0-1.652-.308-2.126-.63a3.32 3.32 0 01-.715-.657l-.014-.02-.005-.006-.002-.003v-.002h-.001l.613-.432-.614.43a.75.75 0 01.183-1.044h.001z"></path></svg>
                  </span>
                </button>
              </span>
              <text-expander keys=": @" data-mention-url="/autocomplete/user-suggestions" data-emoji-url="/autocomplete/emoji">
                <input
                  type="text"
                  autocomplete="off"
                  data-no-org-url="/autocomplete/user-suggestions"
                  data-org-url="/suggestions?mention_suggester=1"
                  data-maxlength="80"
                  class="d-table-cell width-full form-control js-user-status-message-field js-characters-remaining-field"
                  placeholder="What's happening?"
                  name="message"
                  value=""
                  aria-label="What is your current status?">
              </text-expander>
              <div class="error">Could not update your status, please try again.</div>
            </div>
            <div style="margin-left: 53px" class="my-1 text-small label-characters-remaining js-characters-remaining" data-suffix="remaining" hidden>
              80 remaining
            </div>
          </div>
          <include-fragment class="js-user-status-emoji-picker" data-url="/users/status/emoji"></include-fragment>
          <div class="overflow-auto ml-n3 mr-n3 px-3 border-bottom" style="max-height: 33vh">
            <div class="user-status-suggestions js-user-status-suggestions collapsed overflow-hidden">
              <h4 class="f6 text-normal my-3">Suggestions:</h4>
              <div class="mx-3 mt-2 clearfix">
                  <div class="float-left col-6">
                      <button type="button" value=":palm_tree:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="palm_tree" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f334.png">🌴</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          On vacation
                        </div>
                      </button>
                      <button type="button" value=":face_with_thermometer:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="face_with_thermometer" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f912.png">🤒</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Out sick
                        </div>
                      </button>
                  </div>
                  <div class="float-left col-6">
                      <button type="button" value=":house:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="house" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f3e0.png">🏠</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Working from home
                        </div>
                      </button>
                      <button type="button" value=":dart:" class="d-flex flex-items-baseline flex-items-stretch lh-condensed f6 btn-link link-gray no-underline js-predefined-user-status mb-1">
                        <div class="emoji-status-width mr-2 v-align-middle js-predefined-user-status-emoji">
                          <g-emoji alias="dart" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f3af.png">🎯</g-emoji>
                        </div>
                        <div class="d-flex flex-items-center no-underline js-predefined-user-status-message ws-normal text-left" style="border-left: 1px solid transparent">
                          Focusing
                        </div>
                      </button>
                  </div>
              </div>
            </div>
            <div class="user-status-limited-availability-container">
              <div class="form-checkbox my-0">
                <input type="checkbox" name="limited_availability" value="1" class="js-user-status-limited-availability-checkbox" data-default-message="I may be slow to respond." aria-describedby="limited-availability-help-text-truncate-true-compact-true" id="limited-availability-truncate-true-compact-true">
                <label class="d-block f5 text-gray-dark mb-1" for="limited-availability-truncate-true-compact-true">
                  Busy
                </label>
                <p class="note" id="limited-availability-help-text-truncate-true-compact-true">
                  When others mention you, assign you, or request your review,
                  GitHub will let them know that you have limited availability.
                </p>
              </div>
            </div>
          </div>
          <div class="d-inline-block f5 mr-2 pt-3 pb-2" >
  <div class="d-inline-block mr-1">
    Clear status
  </div>

  <details class="js-user-status-expire-drop-down f6 dropdown details-reset details-overlay d-inline-block mr-2">
    <summary class="btn btn-sm v-align-baseline" aria-haspopup="true">
      <div class="js-user-status-expiration-interval-selected d-inline-block v-align-baseline">
        Never
      </div>
      <div class="dropdown-caret"></div>
    </summary>

    <ul class="dropdown-menu dropdown-menu-se pl-0 overflow-auto" style="width: 220px; max-height: 15.5em">
      <li>
        <button type="button" class="btn-link dropdown-item js-user-status-expire-button ws-normal" title="Never">
          <span class="d-inline-block text-bold mb-1">Never</span>
          <div class="f6 lh-condensed">Keep this status until you clear your status or edit your status.</div>
        </button>
      </li>
      <li class="dropdown-divider" role="none"></li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 30 minutes" value="2020-08-15T12:08:11-03:00">
            in 30 minutes
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 1 hour" value="2020-08-15T12:38:11-03:00">
            in 1 hour
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="in 4 hours" value="2020-08-15T15:38:11-03:00">
            in 4 hours
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="today" value="2020-08-15T23:59:59-03:00">
            today
          </button>
        </li>
        <li>
          <button type="button" class="btn-link dropdown-item ws-normal js-user-status-expire-button" title="this week" value="2020-08-16T23:59:59-03:00">
            this week
          </button>
        </li>
    </ul>
  </details>
  <input class="js-user-status-expiration-date-input" type="hidden" name="expires_at" value="">
</div>

          <include-fragment class="js-user-status-org-picker" data-url="/users/status/organizations"></include-fragment>
        </div>
        <div class="d-flex flex-items-center flex-justify-between p-3 border-top">
          <button type="submit" disabled class="width-full btn btn-primary mr-2 js-user-status-submit">
            Set status
          </button>
          <button type="button" disabled class="width-full js-clear-user-status-button btn ml-2 ">
            Clear status
          </button>
        </div>
</form>    </details-dialog>
  </details>
</div>

      </div>
      <div role="none" class="dropdown-divider"></div>

    <a role="menuitem" class="dropdown-item" href="/itfrancisconeto" data-ga-click="Header, go to profile, text:your profile" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;YOUR_PROFILE&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="d8d750bc6f45117086faec8535898fbde88cc765ac08b520dbda8a48c32e9385" >Your profile</a>

    <a role="menuitem" class="dropdown-item" href="/itfrancisconeto?tab=repositories" data-ga-click="Header, go to repositories, text:your repositories" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;YOUR_REPOSITORIES&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="e711235e60c2ca380bf6ae6f9a7b0b976b7cf8484011c97e72bafa475e701b9b" >Your repositories</a>



    <a role="menuitem" class="dropdown-item" href="/itfrancisconeto?tab=projects" data-ga-click="Header, go to projects, text:your projects" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;YOUR_PROJECTS&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="6a5dce19bbcaaa61fae774120020cf612ff2b2349e17a97cca44784908879b09" >Your projects</a>


    <a role="menuitem" class="dropdown-item" href="/itfrancisconeto?tab=stars" data-ga-click="Header, go to starred repos, text:your stars" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;YOUR_STARS&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="bd0ba204d2af1b25a4074e32e0cd9e26294e8abb8b6be73ae3aea6d44bfa2a10" >Your stars</a>
      <a role="menuitem" class="dropdown-item" href="https://gist.github.com/mine" data-ga-click="Header, your gists, text:your gists" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;YOUR_GISTS&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="2d88280dc31c720739747dd8857d6ac18ccccafd7d4c4ece392847efdcaf1373" >Your gists</a>





    <div role="none" class="dropdown-divider"></div>
      <a role="menuitem" class="dropdown-item" href="/settings/billing" data-ga-click="Header, go to billing, text:upgrade" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;UPGRADE&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="000a938d3bb4acb6343e0bc213c9ab14be3a456e8508e31b68c061d7f69953e4" >Upgrade</a>
      
<div id="feature-enrollment-toggle" class="hide-sm hide-md feature-preview-details position-relative">
  <button
    type="button"
    class="dropdown-item btn-link"
    role="menuitem"
    data-feature-preview-trigger-url="/users/itfrancisconeto/feature_previews"
    data-feature-preview-close-details="{&quot;event_type&quot;:&quot;feature_preview.clicks.close_modal&quot;,&quot;payload&quot;:{&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}"
    data-feature-preview-close-hmac="10ae4d3420eed6f3acdda548103235ee7267f190df45c867fab7177497bb79f0"
    data-hydro-click="{&quot;event_type&quot;:&quot;feature_preview.clicks.open_modal&quot;,&quot;payload&quot;:{&quot;link_location&quot;:&quot;user_dropdown&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}"
    data-hydro-click-hmac="4998134c09ada108806da754fa824204959239c84a999ec95a68b5efd69ebb6e"
  >
    Feature preview
  </button>
    <span class="feature-preview-indicator js-feature-preview-indicator" hidden></span>
</div>

    <a role="menuitem" class="dropdown-item" href="https://docs.github.com" data-ga-click="Header, go to help, text:help" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;HELP&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="206541745ad90f2ca92890aae5e3648e9a391bd7d43208e480d1119617c5ce2d" >Help</a>
    <a role="menuitem" class="dropdown-item" href="/settings/profile" data-ga-click="Header, go to settings, icon:settings" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;SETTINGS&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="4a8e1ee2539ea44e7716881e9f004bdc47e29026a826cdfa1345ee5ae1bbb3af" >Settings</a>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="logout-form" action="/logout" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="7S5a+QhfXcZ/jxjteFK8icbHTypkzk5uPYaVfAD21zb80b+4mo/9sXkr1vrSNDVibDui1XsBFLRvOMpqwx2DYA==" />
      
      <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout" data-hydro-click="{&quot;event_type&quot;:&quot;global_header.user_menu_dropdown.click&quot;,&quot;payload&quot;:{&quot;request_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;target&quot;:&quot;SIGN_OUT&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="bc4c879c7fcdcf671260348a61a5dcca7e3a824fd93b75d61e4ad3cfda190435"  role="menuitem">
        Sign out
      </button>
      <input type="text" name="required_field_163c" hidden="hidden" class="form-control" /><input type="hidden" name="timestamp" value="1597502291141" class="form-control" /><input type="hidden" name="timestamp_secret" value="fa7c45442ed426dbbbd8849a55d7920bcc2946283203d7ade9f1bc58d5ea150c" class="form-control" />
</form>  </details-menu>
</details>

  </div>

</header>

          

    </div>

  <div id="start-of-content" class="show-on-focus"></div>




    <div id="js-flash-container">


  <template class="js-flash-template">
    <div class="flash flash-full  js-flash-template-container">
  <div class=" px-2" >
    <button class="flash-close js-flash-close" type="button" aria-label="Dismiss this message">
      <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
    </button>
    
      <div class="js-flash-template-message"></div>

  </div>
</div>
  </template>
</div>


  

  <include-fragment class="js-notification-shelf-include-fragment" data-base-src="https://github.com/notifications/beta/shelf"></include-fragment>



  <div
    class="application-main "
    data-commit-hovercards-enabled
    data-discussion-hovercards-enabled
    data-issue-and-pr-hovercards-enabled
  >
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode" class="">
    <main  >
      

  










  <div class="bg-gray-light pt-3 hide-full-screen mb-5">

    <div class="d-flex mb-3 px-3 px-md-4 px-lg-5">

      <div class="flex-auto min-width-0 width-fit mr-3">
        <h1 class=" d-flex flex-wrap flex-items-center break-word f3 text-normal">
    <svg class="octicon octicon-repo text-gray" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M2 2.5A2.5 2.5 0 014.5 0h8.75a.75.75 0 01.75.75v12.5a.75.75 0 01-.75.75h-2.5a.75.75 0 110-1.5h1.75v-2h-8a1 1 0 00-.714 1.7.75.75 0 01-1.072 1.05A2.495 2.495 0 012 11.5v-9zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 011-1h8zM5 12.25v3.25a.25.25 0 00.4.2l1.45-1.087a.25.25 0 01.3 0L8.6 15.7a.25.25 0 00.4-.2v-3.25a.25.25 0 00-.25-.25h-3.5a.25.25 0 00-.25.25z"></path></svg>
  <span class="author ml-2 flex-self-stretch" itemprop="author">
    <a class="url fn" rel="author" data-hovercard-type="user" data-hovercard-url="/users/itfrancisconeto/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="/itfrancisconeto">itfrancisconeto</a>
  </span>
  <span class="mx-1 flex-self-stretch">/</span>
  <strong itemprop="name" class="mr-2 flex-self-stretch">
    <a data-pjax="#js-repo-pjax-container" href="/itfrancisconeto/PUC_Minas_TCC">PUC_Minas_TCC</a>
  </strong>
  
</h1>


      </div>

      <ul class="pagehead-actions flex-shrink-0 d-none d-md-inline" style="padding: 2px 0;">

  <li>
        <form data-remote="true" class="d-flex js-social-form js-social-container" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="n4RCDouo7+3FZevLZ1CxNdLomTNAYTpUSYynV22LKVx4U6/6yNqoYM0xTnM/h1pxVUxHJDBc95zMrDHD6cArfA==" />      <input type="hidden" name="repository_id" value="264758279">

      <details class="details-reset details-overlay select-menu hx_rsm">
        <summary class="btn btn-sm btn-with-count" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;WATCH_BUTTON&quot;,&quot;repository_id&quot;:264758279,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="aeea8a5c3bbf7dd7ec3ff3ba167807c438776b99cb3c285529a70e69379ea7ca" data-ga-click="Repository, click Watch settings, action:blob#show">          <span data-menu-button>
              <svg height="16" class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
              Unwatch
          </span>
          <span class="dropdown-caret"></span>
</summary>        <details-menu
          class="select-menu-modal position-absolute mt-5"
          style="z-index: 99;">
          <div class="select-menu-header">
            <span class="select-menu-title">Notifications</span>
          </div>
          <div class="select-menu-list">
            <button type="submit" name="do" value="included" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Not watching</span>
                <span class="description">Be notified only when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg height="16" class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
                  Watch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="release_only" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Releases only</span>
                <span class="description">Be notified of new releases, and when participating or @mentioned.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg height="16" class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
                  Unwatch releases
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="subscribed" class="select-menu-item width-full" aria-checked="true" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Watching</span>
                <span class="description">Be notified of all conversations.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg class="octicon octicon-eye v-align-text-bottom" height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.679 7.932c.412-.621 1.242-1.75 2.366-2.717C5.175 4.242 6.527 3.5 8 3.5c1.473 0 2.824.742 3.955 1.715 1.124.967 1.954 2.096 2.366 2.717a.119.119 0 010 .136c-.412.621-1.242 1.75-2.366 2.717C10.825 11.758 9.473 12.5 8 12.5c-1.473 0-2.824-.742-3.955-1.715C2.92 9.818 2.09 8.69 1.679 8.068a.119.119 0 010-.136zM8 2c-1.981 0-3.67.992-4.933 2.078C1.797 5.169.88 6.423.43 7.1a1.619 1.619 0 000 1.798c.45.678 1.367 1.932 2.637 3.024C4.329 13.008 6.019 14 8 14c1.981 0 3.67-.992 4.933-2.078 1.27-1.091 2.187-2.345 2.637-3.023a1.619 1.619 0 000-1.798c-.45-.678-1.367-1.932-2.637-3.023C11.671 2.992 9.981 2 8 2zm0 8a2 2 0 100-4 2 2 0 000 4z"></path></svg>
                  Unwatch
                </span>
              </div>
            </button>

            <button type="submit" name="do" value="ignore" class="select-menu-item width-full" aria-checked="false" role="menuitemradio">
              <svg class="octicon octicon-check select-menu-item-icon" height="16" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>
              <div class="select-menu-item-text">
                <span class="select-menu-item-heading">Ignoring</span>
                <span class="description">Never be notified.</span>
                <span class="hidden-select-button-text" data-menu-button-contents>
                  <svg height="16" class="octicon octicon-mute" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 2.75a.75.75 0 00-1.238-.57L3.472 5H1.75A1.75 1.75 0 000 6.75v2.5C0 10.216.784 11 1.75 11h1.723l3.289 2.82A.75.75 0 008 13.25V2.75zM4.238 6.32L6.5 4.38v7.24L4.238 9.68a.75.75 0 00-.488-.18h-2a.25.25 0 01-.25-.25v-2.5a.25.25 0 01.25-.25h2a.75.75 0 00.488-.18zm7.042-1.1a.75.75 0 10-1.06 1.06L11.94 8l-1.72 1.72a.75.75 0 101.06 1.06L13 9.06l1.72 1.72a.75.75 0 101.06-1.06L14.06 8l1.72-1.72a.75.75 0 00-1.06-1.06L13 6.94l-1.72-1.72z"></path></svg>
                  Stop ignoring
                </span>
              </div>
            </button>
          </div>
        </details-menu>
      </details>
        <a class="social-count js-social-count"
          href="/itfrancisconeto/PUC_Minas_TCC/watchers"
          aria-label="1 user is watching this repository">
          1
        </a>
</form>
  </li>

  <li>
      <div class="js-toggler-container js-social-container starring-container ">
    <form class="starred js-social-form" action="/itfrancisconeto/PUC_Minas_TCC/unstar" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="bFhNHmRw2T1WCFdsk/Di2xcFc9laQOZcVkgnpwLaCHTURzqmlQRA3H9B+BY/rakwNfDQzag9yXmP3qBInRRE6Q==" />
      <input type="hidden" name="context" value="repository"></input>
      <button type="submit" class="btn btn-sm btn-with-count  js-toggler-target" aria-label="Unstar this repository" title="Unstar itfrancisconeto/PUC_Minas_TCC" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;UNSTAR_BUTTON&quot;,&quot;repository_id&quot;:264758279,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="b990e7c61c4249a15591a7e08ede8fe4219384ee6860ca06e51ce3ffae01fe8b" data-ga-click="Repository, click unstar button, action:blob#show; text:Unstar">        <svg height="16" class="octicon octicon-star-fill" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"></path></svg>
        Unstar
</button>        <a class="social-count js-social-count" href="/itfrancisconeto/PUC_Minas_TCC/stargazers"
           aria-label="0 users starred this repository">
           0
        </a>
</form>
    <form class="unstarred js-social-form" action="/itfrancisconeto/PUC_Minas_TCC/star" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="aLGx7ED7mW6kX7yBJy5btLzS5Bwg4pj6FkJ3I/pwgmzEHJJzEQnDqhi/jImWZaDrd7Sy4ujOHhU+NR1rQG28Rg==" />
      <input type="hidden" name="context" value="repository"></input>
      <button type="submit" class="btn btn-sm btn-with-count  js-toggler-target" aria-label="Unstar this repository" title="Star itfrancisconeto/PUC_Minas_TCC" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;STAR_BUTTON&quot;,&quot;repository_id&quot;:264758279,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="9b5425a113719b1d6be4a13272be61c47645a99ab70cf73542b10ef0f9029eca" data-ga-click="Repository, click star button, action:blob#show; text:Star">        <svg height="16" class="octicon octicon-star" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25zm0 2.445L6.615 5.5a.75.75 0 01-.564.41l-3.097.45 2.24 2.184a.75.75 0 01.216.664l-.528 3.084 2.769-1.456a.75.75 0 01.698 0l2.77 1.456-.53-3.084a.75.75 0 01.216-.664l2.24-2.183-3.096-.45a.75.75 0 01-.564-.41L8 2.694v.001z"></path></svg>
        Star
</button>        <a class="social-count js-social-count" href="/itfrancisconeto/PUC_Minas_TCC/stargazers"
           aria-label="0 users starred this repository">
          0
        </a>
</form>  </div>

  </li>

  <li>
        <span class="btn btn-sm btn-with-count disabled tooltipped tooltipped-sw" aria-label="Cannot fork because you own this repository and are not a member of any organizations.">
          <svg class="octicon octicon-repo-forked" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M5 3.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm0 2.122a2.25 2.25 0 10-1.5 0v.878A2.25 2.25 0 005.75 8.5h1.5v2.128a2.251 2.251 0 101.5 0V8.5h1.5a2.25 2.25 0 002.25-2.25v-.878a2.25 2.25 0 10-1.5 0v.878a.75.75 0 01-.75.75h-4.5A.75.75 0 015 6.25v-.878zm3.75 7.378a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm3-8.75a.75.75 0 100-1.5.75.75 0 000 1.5z"></path></svg>
          Fork
</span>
    <a href="/itfrancisconeto/PUC_Minas_TCC/network/members" class="social-count"
       aria-label="0 users forked this repository">
      0
    </a>
  </li>
</ul>

    </div>
    
<nav aria-label="Repository" data-pjax="#js-repo-pjax-container" class="js-repo-nav js-sidenav-container-pjax js-responsive-underlinenav overflow-hidden UnderlineNav px-3 px-md-4 px-lg-5 bg-gray-light">
  <ul class="UnderlineNav-body list-style-none ">
          <li class="d-flex">
        <a class="js-selected-navigation-item selected UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="code-tab" data-hotkey="g c" data-ga-click="Repository, Navigation click, Code tab" aria-current="page" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages repo_deployments /itfrancisconeto/PUC_Minas_TCC" href="/itfrancisconeto/PUC_Minas_TCC">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-code UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 1.06L2.06 8l3.72 3.72a.75.75 0 11-1.06 1.06L.47 8.53a.75.75 0 010-1.06l4.25-4.25zm6.56 0a.75.75 0 10-1.06 1.06L13.94 8l-3.72 3.72a.75.75 0 101.06 1.06l4.25-4.25a.75.75 0 000-1.06l-4.25-4.25z"></path></svg>
            <span data-content="Code">Code</span>
              <span title="Not available" class="Counter "></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="issues-tab" data-hotkey="g i" data-ga-click="Repository, Navigation click, Issues tab" data-selected-links="repo_issues repo_labels repo_milestones /itfrancisconeto/PUC_Minas_TCC/issues" href="/itfrancisconeto/PUC_Minas_TCC/issues">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-issue-opened UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1.5a6.5 6.5 0 100 13 6.5 6.5 0 000-13zM0 8a8 8 0 1116 0A8 8 0 010 8zm9 3a1 1 0 11-2 0 1 1 0 012 0zm-.25-6.25a.75.75 0 00-1.5 0v3.5a.75.75 0 001.5 0v-3.5z"></path></svg>
            <span data-content="Issues">Issues</span>
              <span title="0" hidden="hidden" class="Counter ">0</span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="pull-requests-tab" data-hotkey="g p" data-ga-click="Repository, Navigation click, Pull requests tab" data-selected-links="repo_pulls checks /itfrancisconeto/PUC_Minas_TCC/pulls" href="/itfrancisconeto/PUC_Minas_TCC/pulls">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-git-pull-request UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.177 3.073L9.573.677A.25.25 0 0110 .854v4.792a.25.25 0 01-.427.177L7.177 3.427a.25.25 0 010-.354zM3.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122v5.256a2.251 2.251 0 11-1.5 0V5.372A2.25 2.25 0 011.5 3.25zM11 2.5h-1V4h1a1 1 0 011 1v5.628a2.251 2.251 0 101.5 0V5A2.5 2.5 0 0011 2.5zm1 10.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0zM3.75 12a.75.75 0 100 1.5.75.75 0 000-1.5z"></path></svg>
            <span data-content="Pull requests">Pull requests</span>
              <span title="0" hidden="hidden" class="Counter ">0</span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="actions-tab" data-hotkey="g a" data-ga-click="Repository, Navigation click, Actions tab" data-selected-links="repo_actions /itfrancisconeto/PUC_Minas_TCC/actions" href="/itfrancisconeto/PUC_Minas_TCC/actions">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-play UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zM6.379 5.227A.25.25 0 006 5.442v5.117a.25.25 0 00.379.214l4.264-2.559a.25.25 0 000-.428L6.379 5.227z"></path></svg>
            <span data-content="Actions">Actions</span>
              <span title="Not available" class="Counter "></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="projects-tab" data-hotkey="g b" data-ga-click="Repository, Navigation click, Projects tab" data-selected-links="repo_projects new_repo_project repo_project /itfrancisconeto/PUC_Minas_TCC/projects" href="/itfrancisconeto/PUC_Minas_TCC/projects">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-project UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.75 0A1.75 1.75 0 000 1.75v12.5C0 15.216.784 16 1.75 16h12.5A1.75 1.75 0 0016 14.25V1.75A1.75 1.75 0 0014.25 0H1.75zM1.5 1.75a.25.25 0 01.25-.25h12.5a.25.25 0 01.25.25v12.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25V1.75zM11.75 3a.75.75 0 00-.75.75v7.5a.75.75 0 001.5 0v-7.5a.75.75 0 00-.75-.75zm-8.25.75a.75.75 0 011.5 0v5.5a.75.75 0 01-1.5 0v-5.5zM8 3a.75.75 0 00-.75.75v3.5a.75.75 0 001.5 0v-3.5A.75.75 0 008 3z"></path></svg>
            <span data-content="Projects">Projects</span>
              <span title="0" hidden="hidden" class="Counter ">0</span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="wiki-tab" data-hotkey="g w" data-ga-click="Repository, Navigation click, Wikis tab" data-selected-links="repo_wiki /itfrancisconeto/PUC_Minas_TCC/wiki" href="/itfrancisconeto/PUC_Minas_TCC/wiki">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-book UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm8.755 3a2.25 2.25 0 012.25-2.25H14.5v9h-3.757c-.71 0-1.4.201-1.992.572l.004-7.322zm-1.504 7.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574z"></path></svg>
            <span data-content="Wiki">Wiki</span>
              <span title="Not available" class="Counter "></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="security-tab" data-hotkey="g s" data-ga-click="Repository, Navigation click, Security tab" data-selected-links="security overview alerts policy token_scanning code_scanning /itfrancisconeto/PUC_Minas_TCC/security" href="/itfrancisconeto/PUC_Minas_TCC/security">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-shield UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.467.133a1.75 1.75 0 011.066 0l5.25 1.68A1.75 1.75 0 0115 3.48V7c0 1.566-.32 3.182-1.303 4.682-.983 1.498-2.585 2.813-5.032 3.855a1.7 1.7 0 01-1.33 0c-2.447-1.042-4.049-2.357-5.032-3.855C1.32 10.182 1 8.566 1 7V3.48a1.75 1.75 0 011.217-1.667l5.25-1.68zm.61 1.429a.25.25 0 00-.153 0l-5.25 1.68a.25.25 0 00-.174.238V7c0 1.358.275 2.666 1.057 3.86.784 1.194 2.121 2.34 4.366 3.297a.2.2 0 00.154 0c2.245-.956 3.582-2.104 4.366-3.298C13.225 9.666 13.5 8.36 13.5 7V3.48a.25.25 0 00-.174-.237l-5.25-1.68zM9 10.5a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.75a.75.75 0 10-1.5 0v3a.75.75 0 001.5 0v-3z"></path></svg>
            <span data-content="Security">Security</span>
              <span data-url="/itfrancisconeto/PUC_Minas_TCC/security/overall-count" title="Not available" class="js-security-tab-count Counter "></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="insights-tab" data-ga-click="Repository, Navigation click, Insights tab" data-selected-links="repo_graphs repo_contributors dependency_graph dependabot_updates pulse people /itfrancisconeto/PUC_Minas_TCC/pulse" href="/itfrancisconeto/PUC_Minas_TCC/pulse">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-graph UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 1.75a.75.75 0 00-1.5 0v12.5c0 .414.336.75.75.75h14.5a.75.75 0 000-1.5H1.5V1.75zm14.28 2.53a.75.75 0 00-1.06-1.06L10 7.94 7.53 5.47a.75.75 0 00-1.06 0L3.22 8.72a.75.75 0 001.06 1.06L7 7.06l2.47 2.47a.75.75 0 001.06 0l5.25-5.25z"></path></svg>
            <span data-content="Insights">Insights</span>
              <span title="Not available" class="Counter "></span>
</a>      </li>
      <li class="d-flex">
        <a class="js-selected-navigation-item UnderlineNav-item hx_underlinenav-item no-wrap js-responsive-underlinenav-item" data-tab-item="settings-tab" data-ga-click="Repository, Navigation click, Settings tab" data-selected-links="repo_settings repo_branch_settings hooks integration_installations repo_keys_settings issue_template_editor secrets_settings key_links_settings repo_actions_settings notifications /itfrancisconeto/PUC_Minas_TCC/settings" href="/itfrancisconeto/PUC_Minas_TCC/settings">
              <svg classes="UnderlineNav-octicon" display="none inline" height="16" class="octicon octicon-gear UnderlineNav-octicon d-none d-sm-inline" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.429 1.525a6.593 6.593 0 011.142 0c.036.003.108.036.137.146l.289 1.105c.147.56.55.967.997 1.189.174.086.341.183.501.29.417.278.97.423 1.53.27l1.102-.303c.11-.03.175.016.195.046.219.31.41.641.573.989.014.031.022.11-.059.19l-.815.806c-.411.406-.562.957-.53 1.456a4.588 4.588 0 010 .582c-.032.499.119 1.05.53 1.456l.815.806c.08.08.073.159.059.19a6.494 6.494 0 01-.573.99c-.02.029-.086.074-.195.045l-1.103-.303c-.559-.153-1.112-.008-1.529.27-.16.107-.327.204-.5.29-.449.222-.851.628-.998 1.189l-.289 1.105c-.029.11-.101.143-.137.146a6.613 6.613 0 01-1.142 0c-.036-.003-.108-.037-.137-.146l-.289-1.105c-.147-.56-.55-.967-.997-1.189a4.502 4.502 0 01-.501-.29c-.417-.278-.97-.423-1.53-.27l-1.102.303c-.11.03-.175-.016-.195-.046a6.492 6.492 0 01-.573-.989c-.014-.031-.022-.11.059-.19l.815-.806c.411-.406.562-.957.53-1.456a4.587 4.587 0 010-.582c.032-.499-.119-1.05-.53-1.456l-.815-.806c-.08-.08-.073-.159-.059-.19a6.44 6.44 0 01.573-.99c.02-.029.086-.075.195-.045l1.103.303c.559.153 1.112.008 1.529-.27.16-.107.327-.204.5-.29.449-.222.851-.628.998-1.189l.289-1.105c.029-.11.101-.143.137-.146zM8 0c-.236 0-.47.01-.701.03-.743.065-1.29.615-1.458 1.261l-.29 1.106c-.017.066-.078.158-.211.224a5.994 5.994 0 00-.668.386c-.123.082-.233.09-.3.071L3.27 2.776c-.644-.177-1.392.02-1.82.63a7.977 7.977 0 00-.704 1.217c-.315.675-.111 1.422.363 1.891l.815.806c.05.048.098.147.088.294a6.084 6.084 0 000 .772c.01.147-.038.246-.088.294l-.815.806c-.474.469-.678 1.216-.363 1.891.2.428.436.835.704 1.218.428.609 1.176.806 1.82.63l1.103-.303c.066-.019.176-.011.299.071.213.143.436.272.668.386.133.066.194.158.212.224l.289 1.106c.169.646.715 1.196 1.458 1.26a8.094 8.094 0 001.402 0c.743-.064 1.29-.614 1.458-1.26l.29-1.106c.017-.066.078-.158.211-.224a5.98 5.98 0 00.668-.386c.123-.082.233-.09.3-.071l1.102.302c.644.177 1.392-.02 1.82-.63.268-.382.505-.789.704-1.217.315-.675.111-1.422-.364-1.891l-.814-.806c-.05-.048-.098-.147-.088-.294a6.1 6.1 0 000-.772c-.01-.147.039-.246.088-.294l.814-.806c.475-.469.679-1.216.364-1.891a7.992 7.992 0 00-.704-1.218c-.428-.609-1.176-.806-1.82-.63l-1.103.303c-.066.019-.176.011-.299-.071a5.991 5.991 0 00-.668-.386c-.133-.066-.194-.158-.212-.224L10.16 1.29C9.99.645 9.444.095 8.701.031A8.094 8.094 0 008 0zm1.5 8a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM11 8a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            <span data-content="Settings">Settings</span>
              <span title="Not available" class="Counter "></span>
</a>      </li>

</ul>        <div class="position-absolute right-0 pr-3 pr-md-4 pr-lg-5 js-responsive-underlinenav-overflow" style="visibility:hidden;">
      <details class="details-overlay details-reset position-relative">
  <summary role="button">
              <div class="UnderlineNav-item mr-0 border-0">
            <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path></svg>
            <span class="sr-only">More</span>
          </div>

</summary>            <details-menu role="menu" class="dropdown-menu dropdown-menu-sw ">
  
            <ul>
                <li data-menu-item="code-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC" href="/itfrancisconeto/PUC_Minas_TCC">
                    Code
</a>                </li>
                <li data-menu-item="issues-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/issues" href="/itfrancisconeto/PUC_Minas_TCC/issues">
                    Issues
</a>                </li>
                <li data-menu-item="pull-requests-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/pulls" href="/itfrancisconeto/PUC_Minas_TCC/pulls">
                    Pull requests
</a>                </li>
                <li data-menu-item="actions-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/actions" href="/itfrancisconeto/PUC_Minas_TCC/actions">
                    Actions
</a>                </li>
                <li data-menu-item="projects-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/projects" href="/itfrancisconeto/PUC_Minas_TCC/projects">
                    Projects
</a>                </li>
                <li data-menu-item="wiki-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/wiki" href="/itfrancisconeto/PUC_Minas_TCC/wiki">
                    Wiki
</a>                </li>
                <li data-menu-item="security-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/security" href="/itfrancisconeto/PUC_Minas_TCC/security">
                    Security
</a>                </li>
                <li data-menu-item="insights-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/pulse" href="/itfrancisconeto/PUC_Minas_TCC/pulse">
                    Insights
</a>                </li>
                <li data-menu-item="settings-tab" hidden>
                  <a role="menuitem" class="js-selected-navigation-item dropdown-item" data-selected-links=" /itfrancisconeto/PUC_Minas_TCC/settings" href="/itfrancisconeto/PUC_Minas_TCC/settings">
                    Settings
</a>                </li>
            </ul>

</details-menu>
</details>    </div>

</nav>
  </div>

<div class="container-xl clearfix new-discussion-timeline  px-3 px-md-4 px-lg-5">
  <div class="repository-content " >

    
    
  


    <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/itfrancisconeto/PUC_Minas_TCC/blob/d51d8102a054d091c1c0859ea223c1cc04670256/Script_TCC_Francisco_Neto.py">Permalink</a>

    <!-- blob contrib key: blob_contributors:v22:1601d9642c6b7b3ef9483b7c6bddc9d6 -->
    

    <div class="d-flex flex-items-start flex-shrink-0 pb-3 flex-wrap flex-md-nowrap flex-justify-between flex-md-justify-start">
      
<details class="details-reset details-overlay mr-0 mb-0 " id="branch-select-menu">
  <summary class="btn css-truncate"
           data-hotkey="w"
           title="Switch branches or tags">
    <svg text="gray" height="16" class="octicon octicon-git-branch text-gray" viewBox="0 0 16 16" version="1.1" width="16" aria-hidden="true"><path fill-rule="evenodd" d="M11.75 2.5a.75.75 0 100 1.5.75.75 0 000-1.5zm-2.25.75a2.25 2.25 0 113 2.122V6A2.5 2.5 0 0110 8.5H6a1 1 0 00-1 1v1.128a2.251 2.251 0 11-1.5 0V5.372a2.25 2.25 0 111.5 0v1.836A2.492 2.492 0 016 7h4a1 1 0 001-1v-.628A2.25 2.25 0 019.5 3.25zM4.25 12a.75.75 0 100 1.5.75.75 0 000-1.5zM3.5 3.25a.75.75 0 111.5 0 .75.75 0 01-1.5 0z"></path></svg>
    <span class="css-truncate-target" data-menu-button>master</span>
    <span class="dropdown-caret"></span>
  </summary>

  <details-menu class="SelectMenu SelectMenu--hasFilter" src="/itfrancisconeto/PUC_Minas_TCC/refs/master/Script_TCC_Francisco_Neto.py?source_action=show&amp;source_controller=blob" preload>
    <div class="SelectMenu-modal">
      <include-fragment class="SelectMenu-loading" aria-label="Menu is loading">
        <svg class="octicon octicon-octoface anim-pulse" height="32" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M14.7 5.34c.13-.32.55-1.59-.13-3.31 0 0-1.05-.33-3.44 1.3-1-.28-2.07-.32-3.13-.32s-2.13.04-3.13.32c-2.39-1.64-3.44-1.3-3.44-1.3-.68 1.72-.26 2.99-.13 3.31C.49 6.21 0 7.33 0 8.69 0 13.84 3.33 15 7.98 15S16 13.84 16 8.69c0-1.36-.49-2.48-1.3-3.35zM8 14.02c-3.3 0-5.98-.15-5.98-3.35 0-.76.38-1.48 1.02-2.07 1.07-.98 2.9-.46 4.96-.46 2.07 0 3.88-.52 4.96.46.65.59 1.02 1.3 1.02 2.07 0 3.19-2.68 3.35-5.98 3.35zM5.49 9.01c-.66 0-1.2.8-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.54-1.78-1.2-1.78zm5.02 0c-.66 0-1.2.79-1.2 1.78s.54 1.79 1.2 1.79c.66 0 1.2-.8 1.2-1.79s-.53-1.78-1.2-1.78z"></path></svg>
      </include-fragment>
    </div>
  </details-menu>
</details>

      <h2 id="blob-path" class="breadcrumb flex-auto min-width-0 text-normal mx-0 mx-md-3 width-full width-md-auto flex-order-1 flex-md-order-none mt-3 mt-md-0">
        <span class="js-repo-root text-bold"><span class="js-path-segment d-inline-block wb-break-all"><a data-pjax="true" href="/itfrancisconeto/PUC_Minas_TCC"><span>PUC_Minas_TCC</span></a></span></span><span class="separator">/</span><strong class="final-path">Script_TCC_Francisco_Neto.py</strong>
          <span class="separator">/</span><details class="details-reset details-overlay d-inline" id="jumpto-symbol-select-menu">
  <summary class="btn-link link-gray css-truncate" aria-haspopup="true" data-hotkey="r" data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.click_on_blob_definitions&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;click_on_blob_definitions&quot;,&quot;repository_id&quot;:264758279,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="524395956a2f45330b9c8ab46826c77b335af69032fd23a9d70370146ef6d67a">
      <svg class="octicon octicon-code" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 1.06L2.06 8l3.72 3.72a.75.75 0 11-1.06 1.06L.47 8.53a.75.75 0 010-1.06l4.25-4.25zm6.56 0a.75.75 0 10-1.06 1.06L13.94 8l-3.72 3.72a.75.75 0 101.06 1.06l4.25-4.25a.75.75 0 000-1.06l-4.25-4.25z"></path></svg>
    <span data-menu-button>Jump to</span>
    <span class="dropdown-caret"></span>
  </summary>
  <details-menu class="SelectMenu SelectMenu--hasFilter" role="menu">
    <div class="SelectMenu-modal">
      <header class="SelectMenu-header">
        <span class="SelectMenu-title">Code definitions</span>
        <button class="SelectMenu-closeButton" type="button" data-toggle-for="jumpto-symbol-select-menu">
          <svg aria-label="Close menu" class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" role="img"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
        </button>
      </header>
      <div class="SelectMenu-list">
          <div class="SelectMenu-blankslate">
            <p class="mb-0 text-gray">
              No definitions found in this file.
            </p>
          </div>
        <div data-filterable-for="jumpto-symbols-filter-field" data-filterable-type="substring">
        </div>
      </div>
      <footer class="SelectMenu-footer">
        <div class="d-flex flex-justify-between">
          Code navigation not available for this commit
          <svg class="octicon octicon-dot-fill text-light-gray" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8z"></path></svg>
        </div>
      </footer>
    </div>
  </details-menu>
</details>

      </h2>
      <a href="/itfrancisconeto/PUC_Minas_TCC/find/master"
            class="js-pjax-capture-input btn mr-2 d-none d-md-block"
            data-pjax
            data-hotkey="t">
        Go to file
      </a>

      <details id="blob-more-options-details" class="details-overlay details-reset position-relative">
  <summary role="button">
              <span class="btn">
            <svg aria-label="More options" height="16" class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" version="1.1" width="16" role="img"><path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path></svg>
          </span>

</summary>            <ul class="dropdown-menu dropdown-menu-sw">
            <li class="d-block d-md-none">
              <a class="dropdown-item d-flex flex-items-baseline" data-hydro-click="{&quot;event_type&quot;:&quot;repository.click&quot;,&quot;payload&quot;:{&quot;target&quot;:&quot;FIND_FILE_BUTTON&quot;,&quot;repository_id&quot;:264758279,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}" data-hydro-click-hmac="c697ac333f8bdafcb0d198a9964840e53c8dfc5351c9feab69acaa8d3bb7c09f" data-ga-click="Repository, find file, location:repo overview" data-hotkey="t" data-pjax="true" href="/itfrancisconeto/PUC_Minas_TCC/find/master">
                <span class="flex-auto">Go to file</span>
                <span class="text-small text-gray" aria-hidden="true">T</span>
</a>            </li>
            <li data-toggle-for="blob-more-options-details">
              <button type="button" data-toggle-for="jumpto-line-details-dialog" class="btn-link dropdown-item">
                <span class="d-flex flex-items-baseline">
                  <span class="flex-auto">Go to line</span>
                  <span class="text-small text-gray" aria-hidden="true">L</span>
                </span>
              </button>
            </li>
            <li data-toggle-for="blob-more-options-details">
              <button type="button" data-toggle-for="jumpto-symbol-select-menu" class="btn-link dropdown-item">
                <span class="d-flex flex-items-baseline">
                  <span class="flex-auto">Go to definition</span>
                  <span class="text-small text-gray" aria-hidden="true">R</span>
                </span>
              </button>
            </li>
            <li class="dropdown-divider" role="none"></li>
            <li>
              <clipboard-copy value="Script_TCC_Francisco_Neto.py" class="dropdown-item cursor-pointer" data-toggle-for="blob-more-options-details">
                Copy path
              </clipboard-copy>
            </li>
          </ul>

</details>    </div>



    <div class="Box d-flex flex-column flex-shrink-0 mb-3">
      <include-fragment src="/itfrancisconeto/PUC_Minas_TCC/contributors/master/Script_TCC_Francisco_Neto.py" class="commit-loader">
        <div class="Box-header Box-header--blue d-flex flex-items-center">
          <div class="Skeleton avatar avatar-user flex-shrink-0 ml-n1 mr-n1 mt-n1 mb-n1" style="width:24px;height:24px;"></div>
          <div class="Skeleton Skeleton--text col-5 ml-2">&nbsp;</div>
        </div>

        <div class="Box-body d-flex flex-items-center" >
          <div class="Skeleton Skeleton--text col-1">&nbsp;</div>
          <span class="text-red h6 loader-error">Cannot retrieve contributors at this time</span>
        </div>
</include-fragment>    </div>






    <div class="Box mt-3 position-relative
      ">
      
<div class="Box-header py-2 d-flex flex-column flex-shrink-0 flex-md-row flex-md-items-center">
  <div class="text-mono f6 flex-auto pr-3 flex-order-2 flex-md-order-1 mt-2 mt-md-0">

      636 lines (601 sloc)
      <span class="file-info-divider"></span>
    34.1 KB
  </div>

  <div class="d-flex py-1 py-md-0 flex-auto flex-order-1 flex-md-order-2 flex-sm-grow-0 flex-justify-between">

    <div class="BtnGroup">
      <a href="/itfrancisconeto/PUC_Minas_TCC/raw/master/Script_TCC_Francisco_Neto.py" id="raw-url" role="button" class="btn btn-sm BtnGroup-item ">Raw</a>
        <a href="/itfrancisconeto/PUC_Minas_TCC/blame/master/Script_TCC_Francisco_Neto.py" data-hotkey="b" role="button" class="btn js-update-url-with-hash btn-sm BtnGroup-item ">Blame</a>
    </div>

    <div>
          <a class="btn-octicon tooltipped tooltipped-nw js-remove-unless-platform"
             data-platforms="windows,mac"
             href="x-github-client://openRepo/https://github.com/itfrancisconeto/PUC_Minas_TCC?branch=master&amp;filepath=Script_TCC_Francisco_Neto.py"
             aria-label="Open this file in GitHub Desktop"
             data-ga-click="Repository, open with desktop">
              <svg class="octicon octicon-device-desktop" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.75 2.5h12.5a.25.25 0 01.25.25v7.5a.25.25 0 01-.25.25H1.75a.25.25 0 01-.25-.25v-7.5a.25.25 0 01.25-.25zM14.25 1H1.75A1.75 1.75 0 000 2.75v7.5C0 11.216.784 12 1.75 12h3.727c-.1 1.041-.52 1.872-1.292 2.757A.75.75 0 004.75 16h6.5a.75.75 0 00.565-1.243c-.772-.885-1.193-1.716-1.292-2.757h3.727A1.75 1.75 0 0016 10.25v-7.5A1.75 1.75 0 0014.25 1zM9.018 12H6.982a5.72 5.72 0 01-.765 2.5h3.566a5.72 5.72 0 01-.765-2.5z"></path></svg>
          </a>

          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form js-update-url-with-hash" action="/itfrancisconeto/PUC_Minas_TCC/edit/master/Script_TCC_Francisco_Neto.py" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="pCn3xez3ruhinmm+tXnXCgh7kEwj1kjyhP792845ruf0QUUa9K9VbA0mt21cmx97EHvzWr9nu3Nlsgx04nQxSQ==" />
            <button class="btn-octicon tooltipped tooltipped-nw" type="submit"
              aria-label="Edit this file" data-hotkey="e" data-disable-with>
              <svg class="octicon octicon-pencil" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11.013 1.427a1.75 1.75 0 012.474 0l1.086 1.086a1.75 1.75 0 010 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 01-.927-.928l.929-3.25a1.75 1.75 0 01.445-.758l8.61-8.61zm1.414 1.06a.25.25 0 00-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 000-.354l-1.086-1.086zM11.189 6.25L9.75 4.81l-6.286 6.287a.25.25 0 00-.064.108l-.558 1.953 1.953-.558a.249.249 0 00.108-.064l6.286-6.286z"></path></svg>
            </button>
</form>
          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form" action="/itfrancisconeto/PUC_Minas_TCC/delete/master/Script_TCC_Francisco_Neto.py" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="Y+fhqGVy0PjnwvxSSg95IRZV/zrsx9OFuOs59q3dILZJg+rBq9kWjsFHRIJxhWw41pBGK6K7gUpw3Nn/uav8yw==" />
            <button class="btn-octicon btn-octicon-danger tooltipped tooltipped-nw" type="submit"
              aria-label="Delete this file" data-disable-with>
              <svg class="octicon octicon-trashcan" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M6.5 1.75a.25.25 0 01.25-.25h2.5a.25.25 0 01.25.25V3h-3V1.75zm4.5 0V3h2.25a.75.75 0 010 1.5H2.75a.75.75 0 010-1.5H5V1.75C5 .784 5.784 0 6.75 0h2.5C10.216 0 11 .784 11 1.75zM4.496 6.675a.75.75 0 10-1.492.15l.66 6.6A1.75 1.75 0 005.405 15h5.19c.9 0 1.652-.681 1.741-1.576l.66-6.6a.75.75 0 00-1.492-.149l-.66 6.6a.25.25 0 01-.249.225h-5.19a.25.25 0 01-.249-.225l-.66-6.6z"></path></svg>
            </button>
</form>    </div>
  </div>
</div>


        <div class="js-notice border-bottom p-2">
          <div class="d-flex rounded-1 code-navigation-banner">
            <div class="col-6 pt-4 pl-4 pb-4">
              <div class="d-flex flex-items-center f6">
                <h3 class="mr-2">Code navigation is available!</h3>
              </div>
              <p class="text-gray pt-2">
                Navigate your code with ease. Click on function and method calls to jump to their definitions or references in the same repository.
                <a href="https://docs.github.com/en/articles/navigating-code-on-github">Learn more</a>
              </p>
            </div>
            <div class="col-6 p-2 text-right code-navigation-banner-illo">
              <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-notice-dismiss" action="/settings/dismiss-notice/aleph_code_navigation_banner" accept-charset="UTF-8" method="post"><input type="hidden" name="authenticity_token" value="Hbr19v+3gce0OtZIHDPwpMH7ahDJ324XXOYSDWFnRTwYNIYPO6MnBS7zK5J/KVfiHdOJyhSsjr0mPnrpqCL7Dg==" />
                <button name="button" type="submit" class="btn-link link-gray" aria-label="Dismiss">
                  <svg width="20" height="20" class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
</button></form>            </div>
          </div>
        </div>

      

  <div itemprop="text" class="Box-body p-0 blob-wrapper data type-python  gist-border-0">
      
<table class="highlight tab-size js-file-line-container" data-tab-size="8" data-paste-markdown-skip>
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Importa as bibliotecas necessarias</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>pandas</span> <span class=pl-k>as</span> <span class=pl-s1>pd</span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>matplotlib</span>.<span class=pl-s1>pyplot</span> <span class=pl-k>as</span> <span class=pl-s1>plt</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line"><span class=pl-k>import</span> <span class=pl-s1>seaborn</span> <span class=pl-k>as</span> <span class=pl-s1>sns</span></td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>sklearn</span>.<span class=pl-s1>model_selection</span> <span class=pl-k>import</span> <span class=pl-s1>train_test_split</span></td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>sklearn</span> <span class=pl-k>import</span> <span class=pl-s1>metrics</span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>sklearn</span>.<span class=pl-s1>naive_bayes</span> <span class=pl-k>import</span> <span class=pl-v>GaussianNB</span></td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>sklearn</span>.<span class=pl-s1>ensemble</span> <span class=pl-k>import</span> <span class=pl-v>RandomForestClassifier</span></td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line"><span class=pl-k>from</span> <span class=pl-s1>sklearn</span>.<span class=pl-s1>linear_model</span> <span class=pl-k>import</span> <span class=pl-v>LogisticRegression</span></td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line"><span class=pl-c>################################# FUNÇÕES E VARIÁVEIS DE APOIO #################################</span></td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Variavel global</span></td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>anocorrente</span> <span class=pl-c1>=</span> <span class=pl-c1>2020</span></td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para substituicao de caracteres</span></td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>replacecaracter</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column</span>, <span class=pl-s1>carac_1</span>, <span class=pl-s1>carac_2</span>):</td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-s1>str</span>.<span class=pl-en>replace</span>(<span class=pl-s1>carac_1</span>, <span class=pl-s1>carac_2</span>)</td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para tratamento de valores NaN</span></td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>replacenan</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column</span>):</td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>fillna</span>(<span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para alteracao de tipo de dado para ponto flutuante</span></td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>replacetypetofloat</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column</span>):</td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>astype</span>(<span class=pl-s1>float</span>)</td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>#print(df[column].dtypes)</span></td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para exclusão de linha mediante condição da duração dos rounds ser maior ou igual a 3</span></td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>delrowinvalidround</span>(<span class=pl-s1>df</span>):</td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>column</span> <span class=pl-c1>=</span> <span class=pl-s>&#39;Format&#39;</span>    </td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>key</span>, <span class=pl-s1>value</span> <span class=pl-c1>in</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>iteritems</span>():</td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>item</span> <span class=pl-c1>=</span> <span class=pl-s1>value</span>.<span class=pl-en>split</span>(<span class=pl-s>&#39; &#39;</span>)[<span class=pl-c1>0</span>]        </td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key</span>] <span class=pl-c1>=</span> <span class=pl-en>str</span>(<span class=pl-s1>item</span>)   </td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>df</span>[(<span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>!=</span> <span class=pl-s>&#39;3&#39;</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>!=</span> <span class=pl-s>&#39;5&#39;</span>)].<span class=pl-s1>index</span>)</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-s1>columns</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-s1>columns</span>.<span class=pl-s1>str</span>.<span class=pl-en>replace</span>(<span class=pl-s>&#39;Format&#39;</span>,<span class=pl-s>&#39;Rounds&#39;</span>)    </td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para alteracao de tipo de dado para inteiro</span></td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column</span>):</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>astype</span>(<span class=pl-s1>int</span>)</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>#print(df[column].dtypes)</span></td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para alteracao de tipo de dado para string</span></td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>replacetypetostr</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column</span>):</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>astype</span>(<span class=pl-s1>str</span>)</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>#print(df[column].dtypes)</span></td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para calculo da idade do lutador</span></td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>agefither</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column</span>):</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-s1>str</span>.<span class=pl-en>slice</span>(<span class=pl-s1>start</span><span class=pl-c1>=</span><span class=pl-c1>-</span><span class=pl-c1>4</span>)</td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>fillna</span>(<span class=pl-c1>-</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s1>column</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>astype</span>(<span class=pl-s1>int</span>)</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>key</span>, <span class=pl-s1>value</span> <span class=pl-c1>in</span> <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-en>iteritems</span>(): </td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-s1>value</span> <span class=pl-c1>!=</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>:</td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>value</span> <span class=pl-c1>=</span> <span class=pl-s1>anocorrente</span> <span class=pl-c1>-</span> <span class=pl-s1>value</span></td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>df</span>[<span class=pl-s1>column</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key</span>] <span class=pl-c1>=</span> <span class=pl-s1>value</span>      </td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para plotagem de grafico da frequencia das posturas dos lutadores</span></td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>frequencystance</span>(<span class=pl-s1>df</span>,<span class=pl-s1>stc_open</span>,<span class=pl-s1>stc_orthodox</span>,<span class=pl-s1>stc_sideways</span>,<span class=pl-s1>stc_southpaw</span>,<span class=pl-s1>stc_switch</span>):</td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>grupos</span> <span class=pl-c1>=</span> [<span class=pl-s1>stc_open</span>,<span class=pl-s1>stc_orthodox</span>,<span class=pl-s1>stc_sideways</span>,<span class=pl-s1>stc_southpaw</span>,<span class=pl-s1>stc_switch</span>]</td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>vlr_stc_open</span> <span class=pl-c1>=</span> <span class=pl-en>len</span>(<span class=pl-s1>df</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>df</span>[<span class=pl-s1>stc_open</span>] <span class=pl-c1>==</span> <span class=pl-c1>1</span>])</td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>vlr_stc_orthodox</span> <span class=pl-c1>=</span>  <span class=pl-en>len</span>(<span class=pl-s1>df</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>df</span>[<span class=pl-s1>stc_orthodox</span>] <span class=pl-c1>==</span> <span class=pl-c1>1</span>])</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>vlr_stc_sideways</span> <span class=pl-c1>=</span>  <span class=pl-en>len</span>(<span class=pl-s1>df</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>df</span>[<span class=pl-s1>stc_sideways</span>] <span class=pl-c1>==</span> <span class=pl-c1>1</span>])</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>vlr_stc_southpaw</span> <span class=pl-c1>=</span>  <span class=pl-en>len</span>(<span class=pl-s1>df</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>df</span>[<span class=pl-s1>stc_southpaw</span>] <span class=pl-c1>==</span> <span class=pl-c1>1</span>])</td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>vlr_stc_switch</span> <span class=pl-c1>=</span>  <span class=pl-en>len</span>(<span class=pl-s1>df</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>df</span>[<span class=pl-s1>stc_switch</span>] <span class=pl-c1>==</span> <span class=pl-c1>1</span>])</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>valores</span> <span class=pl-c1>=</span> [<span class=pl-s1>vlr_stc_open</span>,<span class=pl-s1>vlr_stc_orthodox</span>,<span class=pl-s1>vlr_stc_sideways</span>,<span class=pl-s1>vlr_stc_southpaw</span>,<span class=pl-s1>vlr_stc_switch</span>]</td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>figure</span>(<span class=pl-s1>figsize</span><span class=pl-c1>=</span>(<span class=pl-c1>12</span>,<span class=pl-c1>10</span>))</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>title</span>(<span class=pl-s>&#39;Frequencia das posturas&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>xlabel</span>(<span class=pl-s>&#39;Posturas&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>ylabel</span>(<span class=pl-s>&#39;Frequencias&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>bar</span>(<span class=pl-s1>grupos</span>,<span class=pl-s1>valores</span>)</td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;Valores das frequencias: &#39;</span>, <span class=pl-s1>valores</span>)</td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>show</span>()  </td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para plotagem de grafico da correlação entre os atributos dos lutadores</span></td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>figthercorrelation</span>(<span class=pl-s1>df</span>,<span class=pl-s1>height</span>,<span class=pl-s1>weight</span>,<span class=pl-s1>reach</span>,<span class=pl-s1>age</span>,<span class=pl-s1>r_stc_open</span>,<span class=pl-s1>r_stc_orthodox</span>,<span class=pl-s1>r_stc_sideways</span>,<span class=pl-s1>r_stc_southpaw</span>,<span class=pl-s1>r_stc_switch</span>,<span class=pl-s1>b_stc_open</span>,<span class=pl-s1>b_stc_orthodox</span>,<span class=pl-s1>b_stc_sideways</span>,<span class=pl-s1>b_stc_southpaw</span>,<span class=pl-s1>b_stc_switch</span>):</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>height</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>weight</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>reach</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>age</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>r_stc_open</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>r_stc_orthodox</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>r_stc_sideways</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>r_stc_southpaw</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>r_stc_switch</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>b_stc_open</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>b_stc_orthodox</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>b_stc_sideways</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>b_stc_southpaw</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>b_stc_switch</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>figure</span>(<span class=pl-s1>figsize</span><span class=pl-c1>=</span>(<span class=pl-c1>12</span>,<span class=pl-c1>10</span>))</td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_corr_</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>corr</span>()</td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ax_</span> <span class=pl-c1>=</span> <span class=pl-s1>sns</span>.<span class=pl-en>heatmap</span>(<span class=pl-s1>df_corr_</span>, <span class=pl-s1>annot</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>bottom</span>, <span class=pl-s1>top</span> <span class=pl-c1>=</span> <span class=pl-s1>ax_</span>.<span class=pl-en>get_ylim</span>()</td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>ax_</span>.<span class=pl-en>set_ylim</span>(<span class=pl-s1>bottom</span> <span class=pl-c1>+</span> <span class=pl-c1>0.5</span>, <span class=pl-s1>top</span> <span class=pl-c1>-</span> <span class=pl-c1>0.5</span>)</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>show</span>()</td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para plotagem de grafico da frequencia das idades do lutador pela categoria de peso</span></td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>frequencyagebycategory</span>(<span class=pl-s1>df</span>,<span class=pl-s1>column_age</span>,<span class=pl-s1>category</span>):</td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>idx</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>df</span>[<span class=pl-s1>column_age</span>] <span class=pl-c1>==</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>].<span class=pl-s1>index</span></td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>idx</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)    </td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>df</span>[(<span class=pl-s1>df</span>[<span class=pl-s>&#39;Fight_type&#39;</span>] <span class=pl-c1>!=</span> <span class=pl-s1>category</span>)].<span class=pl-s1>index</span>)</td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column_age</span>)</td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>sns</span>.<span class=pl-en>distplot</span>(<span class=pl-s1>df</span>[<span class=pl-s1>column_age</span>], <span class=pl-s1>kde</span><span class=pl-c1>=</span><span class=pl-c1>False</span>, <span class=pl-s1>color</span><span class=pl-c1>=</span><span class=pl-s>&#39;blue&#39;</span>, <span class=pl-s1>bins</span><span class=pl-c1>=</span><span class=pl-c1>25</span>)</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>title</span>(<span class=pl-s>&#39;Frequencia das idades&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>xlabel</span>(<span class=pl-s>&#39;Idades&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>ylabel</span>(<span class=pl-s>&#39;Frequencias&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>show</span>()</td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para plotagem de grafico da frequencia das idades do lutador pela categoria de peso</span></td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>frequencyreachbycategory</span>(<span class=pl-s1>df</span>,<span class=pl-s1>column_reach</span>,<span class=pl-s1>category</span>):</td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>idx</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>df</span>[<span class=pl-s1>column_reach</span>] <span class=pl-c1>==</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>].<span class=pl-s1>index</span></td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>idx</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)    </td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>df</span>[(<span class=pl-s1>df</span>[<span class=pl-s>&#39;Fight_type&#39;</span>] <span class=pl-c1>!=</span> <span class=pl-s1>category</span>)].<span class=pl-s1>index</span>)</td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df</span>, <span class=pl-s1>column_reach</span>)</td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>sns</span>.<span class=pl-en>distplot</span>(<span class=pl-s1>df</span>[<span class=pl-s1>column_reach</span>], <span class=pl-s1>kde</span><span class=pl-c1>=</span><span class=pl-c1>False</span>, <span class=pl-s1>color</span><span class=pl-c1>=</span><span class=pl-s>&#39;blue&#39;</span>, <span class=pl-s1>bins</span><span class=pl-c1>=</span><span class=pl-c1>12</span>)</td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>title</span>(<span class=pl-s>&#39;Frequencia das envergaduras&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>xlabel</span>(<span class=pl-s>&#39;Envergaduras&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>ylabel</span>(<span class=pl-s>&#39;Frequencias&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>show</span>()</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para plotagem de grafico boxplot dos atributos dos lutadures</span></td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df</span>,<span class=pl-s1>column</span>):</td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>figure</span>(<span class=pl-s1>figsize</span><span class=pl-c1>=</span>(<span class=pl-c1>12</span>,<span class=pl-c1>10</span>))</td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>sns</span>.<span class=pl-en>boxplot</span>( <span class=pl-s1>y</span><span class=pl-c1>=</span><span class=pl-s1>df</span>[<span class=pl-s1>column</span>],<span class=pl-s1>width</span><span class=pl-c1>=</span><span class=pl-c1>0.5</span>);</td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>plt</span>.<span class=pl-en>show</span>()</td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para calculo da idade do lutador cuja DOB não foi informado</span></td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>agefithermean</span>(<span class=pl-s1>df</span>,<span class=pl-s1>column_age</span>):    </td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>idx</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>df</span>[<span class=pl-s1>column_age</span>] <span class=pl-c1>==</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>].<span class=pl-s1>index</span></td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>idx</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>groupby</span>(<span class=pl-s1>by</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;Fight_type&#39;</span>])[<span class=pl-s1>column_age</span>].<span class=pl-en>agg</span>(<span class=pl-s1>pd</span>.<span class=pl-v>Series</span>.<span class=pl-s1>mean</span>).<span class=pl-en>to_frame</span>()</td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>idx_to_column</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-s1>index</span>.<span class=pl-s1>values</span></td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>insert</span>(<span class=pl-c1>0</span>,<span class=pl-s1>column</span><span class=pl-c1>=</span><span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s1>value</span> <span class=pl-c1>=</span> <span class=pl-s1>idx_to_column</span>)</td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>reset_index</span>(<span class=pl-s1>drop</span><span class=pl-c1>=</span><span class=pl-c1>True</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para atribuir a media da idade por categoria para os valores de idade missing</span></td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>agefithermissing</span>(<span class=pl-s1>df_orig</span>,<span class=pl-s1>df_dest</span>,<span class=pl-s1>column_orig</span>,<span class=pl-s1>column_dest</span>):</td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>key_dest</span>, <span class=pl-s1>value_dest</span> <span class=pl-c1>in</span> <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;Fight_type&#39;</span>].<span class=pl-en>iteritems</span>():</td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>key_orig</span>, <span class=pl-s1>value_orig</span> <span class=pl-c1>in</span> <span class=pl-s1>df_orig</span>[<span class=pl-s>&#39;Fight_type&#39;</span>].<span class=pl-en>iteritems</span>():</td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> <span class=pl-s1>value_dest</span> <span class=pl-c1>==</span> <span class=pl-s1>value_orig</span>:</td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>idx</span> <span class=pl-c1>=</span> <span class=pl-s1>df_orig</span>[<span class=pl-s1>df_orig</span>[<span class=pl-s>&#39;Fight_type&#39;</span>]<span class=pl-c1>==</span><span class=pl-s1>value_orig</span>].<span class=pl-s1>index</span>.<span class=pl-s1>values</span>.<span class=pl-en>item</span>()</td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>item_orig</span> <span class=pl-c1>=</span> <span class=pl-s1>df_orig</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>idx</span>,[<span class=pl-s1>column_orig</span>]]              </td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>item_dest</span> <span class=pl-c1>=</span> <span class=pl-s1>df_dest</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>key_dest</span>,<span class=pl-s1>column_dest</span>]</td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line">               <span class=pl-k>if</span> <span class=pl-s1>item_dest</span> <span class=pl-c1>==</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>:</td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">                   <span class=pl-s1>df_dest</span>[<span class=pl-s1>column_dest</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key_dest</span>] <span class=pl-c1>=</span> <span class=pl-s1>item_orig</span></td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">               <span class=pl-k>break</span>    </td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df_dest</span> </td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para calculo da enveregadura do lutador cuja Reach não foi informado</span></td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>reachfithermean</span>(<span class=pl-s1>df</span>,<span class=pl-s1>column_reach</span>):    </td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>idx</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s1>df</span>[<span class=pl-s1>column_reach</span>] <span class=pl-c1>==</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>].<span class=pl-s1>index</span></td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s1>idx</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>groupby</span>(<span class=pl-s1>by</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;Fight_type&#39;</span>])[<span class=pl-s1>column_reach</span>].<span class=pl-en>agg</span>(<span class=pl-s1>pd</span>.<span class=pl-v>Series</span>.<span class=pl-s1>mean</span>).<span class=pl-en>to_frame</span>()</td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>idx_to_column</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-s1>index</span>.<span class=pl-s1>values</span></td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>insert</span>(<span class=pl-c1>0</span>,<span class=pl-s1>column</span><span class=pl-c1>=</span><span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s1>value</span> <span class=pl-c1>=</span> <span class=pl-s1>idx_to_column</span>)</td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>reset_index</span>(<span class=pl-s1>drop</span><span class=pl-c1>=</span><span class=pl-c1>True</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para atribuir a media da idade por categoria para os valores de idade missing</span></td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>reachfithermissing</span>(<span class=pl-s1>df_orig</span>,<span class=pl-s1>df_dest</span>,<span class=pl-s1>column_orig</span>,<span class=pl-s1>column_dest</span>):</td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>key_dest</span>, <span class=pl-s1>value_dest</span> <span class=pl-c1>in</span> <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;Fight_type&#39;</span>].<span class=pl-en>iteritems</span>():</td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>key_orig</span>, <span class=pl-s1>value_orig</span> <span class=pl-c1>in</span> <span class=pl-s1>df_orig</span>[<span class=pl-s>&#39;Fight_type&#39;</span>].<span class=pl-en>iteritems</span>():</td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> <span class=pl-s1>value_dest</span> <span class=pl-c1>==</span> <span class=pl-s1>value_orig</span>:</td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>idx</span> <span class=pl-c1>=</span> <span class=pl-s1>df_orig</span>[<span class=pl-s1>df_orig</span>[<span class=pl-s>&#39;Fight_type&#39;</span>]<span class=pl-c1>==</span><span class=pl-s1>value_orig</span>].<span class=pl-s1>index</span>.<span class=pl-s1>values</span>.<span class=pl-en>item</span>()</td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>item_orig</span> <span class=pl-c1>=</span> <span class=pl-s1>df_orig</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>idx</span>,[<span class=pl-s1>column_orig</span>]]</td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>item_dest</span> <span class=pl-c1>=</span> <span class=pl-s1>df_dest</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>key_dest</span>,<span class=pl-s1>column_dest</span>]</td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line">               <span class=pl-k>if</span> <span class=pl-s1>item_dest</span> <span class=pl-c1>==</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>:</td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line">                   <span class=pl-s1>df_dest</span>[<span class=pl-s1>column_dest</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key_dest</span>] <span class=pl-c1>=</span> <span class=pl-s1>item_orig</span></td>
      </tr>
      <tr>
        <td id="L178" class="blob-num js-line-number" data-line-number="178"></td>
        <td id="LC178" class="blob-code blob-code-inner js-file-line">               <span class=pl-k>break</span></td>
      </tr>
      <tr>
        <td id="L179" class="blob-num js-line-number" data-line-number="179"></td>
        <td id="LC179" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df_dest</span></td>
      </tr>
      <tr>
        <td id="L180" class="blob-num js-line-number" data-line-number="180"></td>
        <td id="LC180" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L181" class="blob-num js-line-number" data-line-number="181"></td>
        <td id="LC181" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para popular atributos do merge dos dataframe</span></td>
      </tr>
      <tr>
        <td id="L182" class="blob-num js-line-number" data-line-number="182"></td>
        <td id="LC182" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>,<span class=pl-s1>df_dest</span>,<span class=pl-s1>column_orig</span>,<span class=pl-s1>column_dest</span>,<span class=pl-s1>fighter_dest</span>):    </td>
      </tr>
      <tr>
        <td id="L183" class="blob-num js-line-number" data-line-number="183"></td>
        <td id="LC183" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>key_dest</span>, <span class=pl-s1>value_dest</span> <span class=pl-c1>in</span> <span class=pl-s1>df_dest</span>[<span class=pl-s1>fighter_dest</span>].<span class=pl-en>iteritems</span>():</td>
      </tr>
      <tr>
        <td id="L184" class="blob-num js-line-number" data-line-number="184"></td>
        <td id="LC184" class="blob-code blob-code-inner js-file-line">        <span class=pl-k>for</span> <span class=pl-s1>key_orig</span>, <span class=pl-s1>value_orig</span> <span class=pl-c1>in</span> <span class=pl-s1>df_orig</span>[<span class=pl-s>&#39;fighter_name&#39;</span>].<span class=pl-en>iteritems</span>():</td>
      </tr>
      <tr>
        <td id="L185" class="blob-num js-line-number" data-line-number="185"></td>
        <td id="LC185" class="blob-code blob-code-inner js-file-line">            <span class=pl-k>if</span> <span class=pl-s1>value_dest</span> <span class=pl-c1>==</span> <span class=pl-s1>value_orig</span>:</td>
      </tr>
      <tr>
        <td id="L186" class="blob-num js-line-number" data-line-number="186"></td>
        <td id="LC186" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>idx</span> <span class=pl-c1>=</span> <span class=pl-s1>df_orig</span>[<span class=pl-s1>df_orig</span>[<span class=pl-s>&#39;fighter_name&#39;</span>]<span class=pl-c1>==</span><span class=pl-s1>value_orig</span>].<span class=pl-s1>index</span>.<span class=pl-s1>values</span>.<span class=pl-en>item</span>()</td>
      </tr>
      <tr>
        <td id="L187" class="blob-num js-line-number" data-line-number="187"></td>
        <td id="LC187" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>item_orig</span> <span class=pl-c1>=</span> <span class=pl-s1>df_orig</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>idx</span>,[<span class=pl-s1>column_orig</span>]]              </td>
      </tr>
      <tr>
        <td id="L188" class="blob-num js-line-number" data-line-number="188"></td>
        <td id="LC188" class="blob-code blob-code-inner js-file-line">               <span class=pl-s1>df_dest</span>[<span class=pl-s1>column_dest</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key_dest</span>] <span class=pl-c1>=</span> <span class=pl-s1>item_orig</span></td>
      </tr>
      <tr>
        <td id="L189" class="blob-num js-line-number" data-line-number="189"></td>
        <td id="LC189" class="blob-code blob-code-inner js-file-line">               <span class=pl-k>break</span>    </td>
      </tr>
      <tr>
        <td id="L190" class="blob-num js-line-number" data-line-number="190"></td>
        <td id="LC190" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df_dest</span></td>
      </tr>
      <tr>
        <td id="L191" class="blob-num js-line-number" data-line-number="191"></td>
        <td id="LC191" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L192" class="blob-num js-line-number" data-line-number="192"></td>
        <td id="LC192" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para merge dos dataframe</span></td>
      </tr>
      <tr>
        <td id="L193" class="blob-num js-line-number" data-line-number="193"></td>
        <td id="LC193" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>mergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>):</td>
      </tr>
      <tr>
        <td id="L194" class="blob-num js-line-number" data-line-number="194"></td>
        <td id="LC194" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Height&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0.0</span></td>
      </tr>
      <tr>
        <td id="L195" class="blob-num js-line-number" data-line-number="195"></td>
        <td id="LC195" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Weight&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L196" class="blob-num js-line-number" data-line-number="196"></td>
        <td id="LC196" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Reach&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L197" class="blob-num js-line-number" data-line-number="197"></td>
        <td id="LC197" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Age&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L198" class="blob-num js-line-number" data-line-number="198"></td>
        <td id="LC198" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Stance_Open&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L199" class="blob-num js-line-number" data-line-number="199"></td>
        <td id="LC199" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Stance_Orthodox&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L200" class="blob-num js-line-number" data-line-number="200"></td>
        <td id="LC200" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Stance_Sideways&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L201" class="blob-num js-line-number" data-line-number="201"></td>
        <td id="LC201" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Stance_Southpaw&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L202" class="blob-num js-line-number" data-line-number="202"></td>
        <td id="LC202" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;R_Stance_Switch&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L203" class="blob-num js-line-number" data-line-number="203"></td>
        <td id="LC203" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Height&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0.0</span></td>
      </tr>
      <tr>
        <td id="L204" class="blob-num js-line-number" data-line-number="204"></td>
        <td id="LC204" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Weight&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L205" class="blob-num js-line-number" data-line-number="205"></td>
        <td id="LC205" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Reach&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L206" class="blob-num js-line-number" data-line-number="206"></td>
        <td id="LC206" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Age&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L207" class="blob-num js-line-number" data-line-number="207"></td>
        <td id="LC207" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Stance_Open&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L208" class="blob-num js-line-number" data-line-number="208"></td>
        <td id="LC208" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Stance_Orthodox&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L209" class="blob-num js-line-number" data-line-number="209"></td>
        <td id="LC209" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Stance_Sideways&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L210" class="blob-num js-line-number" data-line-number="210"></td>
        <td id="LC210" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Stance_Southpaw&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span></td>
      </tr>
      <tr>
        <td id="L211" class="blob-num js-line-number" data-line-number="211"></td>
        <td id="LC211" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span>[<span class=pl-s>&#39;B_Stance_Switch&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span>    </td>
      </tr>
      <tr>
        <td id="L212" class="blob-num js-line-number" data-line-number="212"></td>
        <td id="LC212" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Height</span></td>
      </tr>
      <tr>
        <td id="L213" class="blob-num js-line-number" data-line-number="213"></td>
        <td id="LC213" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 1/18 &gt;&gt;&gt; Populando atributo R_Height no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L214" class="blob-num js-line-number" data-line-number="214"></td>
        <td id="LC214" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Height&#39;</span>, <span class=pl-s>&#39;R_Height&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)      </td>
      </tr>
      <tr>
        <td id="L215" class="blob-num js-line-number" data-line-number="215"></td>
        <td id="LC215" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Weight</span></td>
      </tr>
      <tr>
        <td id="L216" class="blob-num js-line-number" data-line-number="216"></td>
        <td id="LC216" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 2/18 &gt;&gt;&gt; Populando atributo R_Weight no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L217" class="blob-num js-line-number" data-line-number="217"></td>
        <td id="LC217" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Weight&#39;</span>, <span class=pl-s>&#39;R_Weight&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L218" class="blob-num js-line-number" data-line-number="218"></td>
        <td id="LC218" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Reach</span></td>
      </tr>
      <tr>
        <td id="L219" class="blob-num js-line-number" data-line-number="219"></td>
        <td id="LC219" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 3/18 &gt;&gt;&gt; Populando atributo R_Reach no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L220" class="blob-num js-line-number" data-line-number="220"></td>
        <td id="LC220" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Reach&#39;</span>, <span class=pl-s>&#39;R_Reach&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)           </td>
      </tr>
      <tr>
        <td id="L221" class="blob-num js-line-number" data-line-number="221"></td>
        <td id="LC221" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Age</span></td>
      </tr>
      <tr>
        <td id="L222" class="blob-num js-line-number" data-line-number="222"></td>
        <td id="LC222" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 4/18 &gt;&gt;&gt; Populando atributo R_Age no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L223" class="blob-num js-line-number" data-line-number="223"></td>
        <td id="LC223" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>,<span class=pl-s1>df_dest</span>,<span class=pl-s>&#39;DOB&#39;</span>,<span class=pl-s>&#39;R_Age&#39;</span>,<span class=pl-s>&#39;R_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L224" class="blob-num js-line-number" data-line-number="224"></td>
        <td id="LC224" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Stance_Open</span></td>
      </tr>
      <tr>
        <td id="L225" class="blob-num js-line-number" data-line-number="225"></td>
        <td id="LC225" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 5/18 &gt;&gt;&gt; Populando atributo R_Stance_Open no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L226" class="blob-num js-line-number" data-line-number="226"></td>
        <td id="LC226" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Open Stance&#39;</span>, <span class=pl-s>&#39;R_Stance_Open&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L227" class="blob-num js-line-number" data-line-number="227"></td>
        <td id="LC227" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Stance_Orthodox</span></td>
      </tr>
      <tr>
        <td id="L228" class="blob-num js-line-number" data-line-number="228"></td>
        <td id="LC228" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 6/18 &gt;&gt;&gt; Populando atributo R_Stance_Orthodox no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L229" class="blob-num js-line-number" data-line-number="229"></td>
        <td id="LC229" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Orthodox&#39;</span>, <span class=pl-s>&#39;R_Stance_Orthodox&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L230" class="blob-num js-line-number" data-line-number="230"></td>
        <td id="LC230" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Stance_Sideway</span></td>
      </tr>
      <tr>
        <td id="L231" class="blob-num js-line-number" data-line-number="231"></td>
        <td id="LC231" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 7/18 &gt;&gt;&gt; Populando atributo R_Stance_Sideways no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L232" class="blob-num js-line-number" data-line-number="232"></td>
        <td id="LC232" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Sideways&#39;</span>, <span class=pl-s>&#39;R_Stance_Sideways&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L233" class="blob-num js-line-number" data-line-number="233"></td>
        <td id="LC233" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Stance_Southpaw</span></td>
      </tr>
      <tr>
        <td id="L234" class="blob-num js-line-number" data-line-number="234"></td>
        <td id="LC234" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 8/18 &gt;&gt;&gt; Populando atributo R_Stance_Southpaw no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L235" class="blob-num js-line-number" data-line-number="235"></td>
        <td id="LC235" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Southpaw&#39;</span>, <span class=pl-s>&#39;R_Stance_Southpaw&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L236" class="blob-num js-line-number" data-line-number="236"></td>
        <td id="LC236" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo R_Stance_Switch</span></td>
      </tr>
      <tr>
        <td id="L237" class="blob-num js-line-number" data-line-number="237"></td>
        <td id="LC237" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 9/18 &gt;&gt;&gt; Populando atributo R_Stance_Switch no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L238" class="blob-num js-line-number" data-line-number="238"></td>
        <td id="LC238" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Switch&#39;</span>, <span class=pl-s>&#39;R_Stance_Switch&#39;</span>, <span class=pl-s>&#39;R_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L239" class="blob-num js-line-number" data-line-number="239"></td>
        <td id="LC239" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Height</span></td>
      </tr>
      <tr>
        <td id="L240" class="blob-num js-line-number" data-line-number="240"></td>
        <td id="LC240" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 10/18 &gt;&gt;&gt; Populando atributo B_Height no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L241" class="blob-num js-line-number" data-line-number="241"></td>
        <td id="LC241" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Height&#39;</span>, <span class=pl-s>&#39;B_Height&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L242" class="blob-num js-line-number" data-line-number="242"></td>
        <td id="LC242" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Weight</span></td>
      </tr>
      <tr>
        <td id="L243" class="blob-num js-line-number" data-line-number="243"></td>
        <td id="LC243" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 11/18 &gt;&gt;&gt; Populando atributo B_Weight no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L244" class="blob-num js-line-number" data-line-number="244"></td>
        <td id="LC244" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Weight&#39;</span>, <span class=pl-s>&#39;B_Weight&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L245" class="blob-num js-line-number" data-line-number="245"></td>
        <td id="LC245" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Reach</span></td>
      </tr>
      <tr>
        <td id="L246" class="blob-num js-line-number" data-line-number="246"></td>
        <td id="LC246" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 12/18 &gt;&gt;&gt; Populando atributo B_Reach no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L247" class="blob-num js-line-number" data-line-number="247"></td>
        <td id="LC247" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Reach&#39;</span>, <span class=pl-s>&#39;B_Reach&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)           </td>
      </tr>
      <tr>
        <td id="L248" class="blob-num js-line-number" data-line-number="248"></td>
        <td id="LC248" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Age</span></td>
      </tr>
      <tr>
        <td id="L249" class="blob-num js-line-number" data-line-number="249"></td>
        <td id="LC249" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 13/18 &gt;&gt;&gt; Populando atributo B_Age no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L250" class="blob-num js-line-number" data-line-number="250"></td>
        <td id="LC250" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;DOB&#39;</span>, <span class=pl-s>&#39;B_Age&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L251" class="blob-num js-line-number" data-line-number="251"></td>
        <td id="LC251" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Stance_Open</span></td>
      </tr>
      <tr>
        <td id="L252" class="blob-num js-line-number" data-line-number="252"></td>
        <td id="LC252" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 14/18 &gt;&gt;&gt; Populando atributo B_Stance_Open no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L253" class="blob-num js-line-number" data-line-number="253"></td>
        <td id="LC253" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Open Stance&#39;</span>, <span class=pl-s>&#39;B_Stance_Open&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L254" class="blob-num js-line-number" data-line-number="254"></td>
        <td id="LC254" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Stance_Orthodox</span></td>
      </tr>
      <tr>
        <td id="L255" class="blob-num js-line-number" data-line-number="255"></td>
        <td id="LC255" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 15/18 &gt;&gt;&gt; Populando atributo B_Stance_Orthodox no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L256" class="blob-num js-line-number" data-line-number="256"></td>
        <td id="LC256" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Orthodox&#39;</span>, <span class=pl-s>&#39;B_Stance_Orthodox&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L257" class="blob-num js-line-number" data-line-number="257"></td>
        <td id="LC257" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Stance_Sideway</span></td>
      </tr>
      <tr>
        <td id="L258" class="blob-num js-line-number" data-line-number="258"></td>
        <td id="LC258" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 16/18 &gt;&gt;&gt; Populando atributo B_Stance_Sideways no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L259" class="blob-num js-line-number" data-line-number="259"></td>
        <td id="LC259" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Sideways&#39;</span>, <span class=pl-s>&#39;B_Stance_Sideways&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L260" class="blob-num js-line-number" data-line-number="260"></td>
        <td id="LC260" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Stance_Southpaw</span></td>
      </tr>
      <tr>
        <td id="L261" class="blob-num js-line-number" data-line-number="261"></td>
        <td id="LC261" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 17/18 &gt;&gt;&gt; Populando atributo B_Stance_Southpaw no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L262" class="blob-num js-line-number" data-line-number="262"></td>
        <td id="LC262" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Southpaw&#39;</span>, <span class=pl-s>&#39;B_Stance_Southpaw&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L263" class="blob-num js-line-number" data-line-number="263"></td>
        <td id="LC263" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Popula atributo B_Stance_Switch</span></td>
      </tr>
      <tr>
        <td id="L264" class="blob-num js-line-number" data-line-number="264"></td>
        <td id="LC264" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> Processando 18/18 &gt;&gt;&gt; Populando atributo B_Stance_Switch no dataframe de destino&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L265" class="blob-num js-line-number" data-line-number="265"></td>
        <td id="LC265" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df_dest</span> <span class=pl-c1>=</span> <span class=pl-en>populamergedataframe</span>(<span class=pl-s1>df_orig</span>, <span class=pl-s1>df_dest</span>, <span class=pl-s>&#39;Stance_Switch&#39;</span>, <span class=pl-s>&#39;B_Stance_Switch&#39;</span>, <span class=pl-s>&#39;B_fighter&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L266" class="blob-num js-line-number" data-line-number="266"></td>
        <td id="LC266" class="blob-code blob-code-inner js-file-line">    <span class=pl-c># Retorno da função</span></td>
      </tr>
      <tr>
        <td id="L267" class="blob-num js-line-number" data-line-number="267"></td>
        <td id="LC267" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df_dest</span></td>
      </tr>
      <tr>
        <td id="L268" class="blob-num js-line-number" data-line-number="268"></td>
        <td id="LC268" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L269" class="blob-num js-line-number" data-line-number="269"></td>
        <td id="LC269" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para preparar o dataframe para ML</span></td>
      </tr>
      <tr>
        <td id="L270" class="blob-num js-line-number" data-line-number="270"></td>
        <td id="LC270" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>preparedftoml</span>(<span class=pl-s1>df</span>):</td>
      </tr>
      <tr>
        <td id="L271" class="blob-num js-line-number" data-line-number="271"></td>
        <td id="LC271" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_fighter&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L272" class="blob-num js-line-number" data-line-number="272"></td>
        <td id="LC272" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_fighter&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L273" class="blob-num js-line-number" data-line-number="273"></td>
        <td id="LC273" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;Fight_type&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L274" class="blob-num js-line-number" data-line-number="274"></td>
        <td id="LC274" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s>&#39;Target&#39;</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span> </td>
      </tr>
      <tr>
        <td id="L275" class="blob-num js-line-number" data-line-number="275"></td>
        <td id="LC275" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df</span>,<span class=pl-s>&#39;Rounds&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L276" class="blob-num js-line-number" data-line-number="276"></td>
        <td id="LC276" class="blob-code blob-code-inner js-file-line">    <span class=pl-c>#df.info()    </span></td>
      </tr>
      <tr>
        <td id="L277" class="blob-num js-line-number" data-line-number="277"></td>
        <td id="LC277" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>[<span class=pl-s>&#39;Target&#39;</span>] <span class=pl-c1>=</span> <span class=pl-s1>df</span>[<span class=pl-s>&#39;Rounds&#39;</span>] <span class=pl-c1>-</span> <span class=pl-s1>df</span>[<span class=pl-s>&#39;last_round&#39;</span>]    </td>
      </tr>
      <tr>
        <td id="L278" class="blob-num js-line-number" data-line-number="278"></td>
        <td id="LC278" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>key</span>, <span class=pl-s1>value</span> <span class=pl-c1>in</span> <span class=pl-s1>df</span>[<span class=pl-s>&#39;Target&#39;</span>].<span class=pl-en>iteritems</span>(): </td>
      </tr>
      <tr>
        <td id="L279" class="blob-num js-line-number" data-line-number="279"></td>
        <td id="LC279" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>if</span> <span class=pl-s1>value</span> <span class=pl-c1>==</span> <span class=pl-c1>0</span>:</td>
      </tr>
      <tr>
        <td id="L280" class="blob-num js-line-number" data-line-number="280"></td>
        <td id="LC280" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>df</span>[<span class=pl-s>&#39;Target&#39;</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key</span>] <span class=pl-c1>=</span> <span class=pl-c1>1</span></td>
      </tr>
      <tr>
        <td id="L281" class="blob-num js-line-number" data-line-number="281"></td>
        <td id="LC281" class="blob-code blob-code-inner js-file-line">      <span class=pl-k>else</span>:</td>
      </tr>
      <tr>
        <td id="L282" class="blob-num js-line-number" data-line-number="282"></td>
        <td id="LC282" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>df</span>[<span class=pl-s>&#39;Target&#39;</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key</span>] <span class=pl-c1>=</span> <span class=pl-c1>0</span>          </td>
      </tr>
      <tr>
        <td id="L283" class="blob-num js-line-number" data-line-number="283"></td>
        <td id="LC283" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>return</span> <span class=pl-s1>df</span></td>
      </tr>
      <tr>
        <td id="L284" class="blob-num js-line-number" data-line-number="284"></td>
        <td id="LC284" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L285" class="blob-num js-line-number" data-line-number="285"></td>
        <td id="LC285" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Funcao para exibir a quantidade de lutadore por categoria de peso</span></td>
      </tr>
      <tr>
        <td id="L286" class="blob-num js-line-number" data-line-number="286"></td>
        <td id="LC286" class="blob-code blob-code-inner js-file-line"><span class=pl-k>def</span> <span class=pl-en>countfigherbycategory</span>(<span class=pl-s1>df</span>):</td>
      </tr>
      <tr>
        <td id="L287" class="blob-num js-line-number" data-line-number="287"></td>
        <td id="LC287" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>filter</span>([<span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s>&#39;R_fighter&#39;</span>],<span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L288" class="blob-num js-line-number" data-line-number="288"></td>
        <td id="LC288" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-s1>columns</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-s1>columns</span>.<span class=pl-s1>str</span>.<span class=pl-en>replace</span>(<span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s>&#39;Category&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L289" class="blob-num js-line-number" data-line-number="289"></td>
        <td id="LC289" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span>.<span class=pl-s1>columns</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-s1>columns</span>.<span class=pl-s1>str</span>.<span class=pl-en>replace</span>(<span class=pl-s>&#39;R_fighter&#39;</span>,<span class=pl-s>&#39;QtdFither&#39;</span>)    </td>
      </tr>
      <tr>
        <td id="L290" class="blob-num js-line-number" data-line-number="290"></td>
        <td id="LC290" class="blob-code blob-code-inner js-file-line">    <span class=pl-k>for</span> <span class=pl-s1>key</span>, <span class=pl-s1>value</span> <span class=pl-c1>in</span> <span class=pl-s1>df</span>[<span class=pl-s>&#39;QtdFither&#39;</span>].<span class=pl-en>iteritems</span>():        </td>
      </tr>
      <tr>
        <td id="L291" class="blob-num js-line-number" data-line-number="291"></td>
        <td id="LC291" class="blob-code blob-code-inner js-file-line">        <span class=pl-s1>df</span>[<span class=pl-s>&#39;QtdFither&#39;</span>].<span class=pl-s1>at</span>[<span class=pl-s1>key</span>] <span class=pl-c1>=</span> <span class=pl-c1>1</span>      </td>
      </tr>
      <tr>
        <td id="L292" class="blob-num js-line-number" data-line-number="292"></td>
        <td id="LC292" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>groupby</span>(<span class=pl-s1>by</span><span class=pl-c1>=</span>[<span class=pl-s>&#39;Category&#39;</span>])[<span class=pl-s>&#39;QtdFither&#39;</span>].<span class=pl-en>agg</span>(<span class=pl-s1>sum</span>).<span class=pl-en>to_frame</span>()</td>
      </tr>
      <tr>
        <td id="L293" class="blob-num js-line-number" data-line-number="293"></td>
        <td id="LC293" class="blob-code blob-code-inner js-file-line">    <span class=pl-s1>df</span> <span class=pl-c1>=</span> <span class=pl-s1>df</span>.<span class=pl-en>reset_index</span>()</td>
      </tr>
      <tr>
        <td id="L294" class="blob-num js-line-number" data-line-number="294"></td>
        <td id="LC294" class="blob-code blob-code-inner js-file-line">    <span class=pl-en>print</span>(<span class=pl-s1>df</span>.<span class=pl-en>sort_values</span>(<span class=pl-s1>by</span><span class=pl-c1>=</span><span class=pl-s>&#39;QtdFither&#39;</span>, <span class=pl-s1>ascending</span><span class=pl-c1>=</span><span class=pl-c1>False</span>))</td>
      </tr>
      <tr>
        <td id="L295" class="blob-num js-line-number" data-line-number="295"></td>
        <td id="LC295" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L296" class="blob-num js-line-number" data-line-number="296"></td>
        <td id="LC296" class="blob-code blob-code-inner js-file-line"><span class=pl-c>################################# INICIO DO PROCESSO #################################   </span></td>
      </tr>
      <tr>
        <td id="L297" class="blob-num js-line-number" data-line-number="297"></td>
        <td id="LC297" class="blob-code blob-code-inner js-file-line">    </td>
      </tr>
      <tr>
        <td id="L298" class="blob-num js-line-number" data-line-number="298"></td>
        <td id="LC298" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### IMPORTAÇÃO DO DATAFRAME COM DETALHES DOS LUTADORES (raw_fighter_details)</span></td>
      </tr>
      <tr>
        <td id="L299" class="blob-num js-line-number" data-line-number="299"></td>
        <td id="LC299" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> INICIO DO PROCESSO&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L300" class="blob-num js-line-number" data-line-number="300"></td>
        <td id="LC300" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 1) Importa para dataframe o arquivo com detalhes dos lutadores&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L301" class="blob-num js-line-number" data-line-number="301"></td>
        <td id="LC301" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Faz a leitura do arquivo CSV</span></td>
      </tr>
      <tr>
        <td id="L302" class="blob-num js-line-number" data-line-number="302"></td>
        <td id="LC302" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_1_fighter</span> <span class=pl-c1>=</span> <span class=pl-s1>pd</span>.<span class=pl-en>read_csv</span>(<span class=pl-s>&quot;Dataset/raw_fighter_details.csv&quot;</span>,<span class=pl-s1>encoding</span><span class=pl-c1>=</span><span class=pl-s>&quot;ISO-8859-1&quot;</span>)</td>
      </tr>
      <tr>
        <td id="L303" class="blob-num js-line-number" data-line-number="303"></td>
        <td id="LC303" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Faz uma copia do dataframe original</span></td>
      </tr>
      <tr>
        <td id="L304" class="blob-num js-line-number" data-line-number="304"></td>
        <td id="LC304" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_1_fighter</span>.<span class=pl-en>copy</span>()</td>
      </tr>
      <tr>
        <td id="L305" class="blob-num js-line-number" data-line-number="305"></td>
        <td id="LC305" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L306" class="blob-num js-line-number" data-line-number="306"></td>
        <td id="LC306" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Verifica a existencia de dados faltantes</span></td>
      </tr>
      <tr>
        <td id="L307" class="blob-num js-line-number" data-line-number="307"></td>
        <td id="LC307" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 2) Verifica a existencia de dados faltantes em df_2_fighter_proc:&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L308" class="blob-num js-line-number" data-line-number="308"></td>
        <td id="LC308" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s1>df_2_fighter_proc</span>.<span class=pl-en>isnull</span>().<span class=pl-en>sum</span>())</td>
      </tr>
      <tr>
        <td id="L309" class="blob-num js-line-number" data-line-number="309"></td>
        <td id="LC309" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L310" class="blob-num js-line-number" data-line-number="310"></td>
        <td id="LC310" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L311" class="blob-num js-line-number" data-line-number="311"></td>
        <td id="LC311" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### SUBSTITUICAO DE CARACTERES E TIPOS DE DADOS</span></td>
      </tr>
      <tr>
        <td id="L312" class="blob-num js-line-number" data-line-number="312"></td>
        <td id="LC312" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 3) Substitui caracteres e tipos de dados no dataframe&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L313" class="blob-num js-line-number" data-line-number="313"></td>
        <td id="LC313" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Substitui caracteres da coluna Height (Altura)</span></td>
      </tr>
      <tr>
        <td id="L314" class="blob-num js-line-number" data-line-number="314"></td>
        <td id="LC314" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacecaracter</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Height&#39;</span>,<span class=pl-s>&#39;<span class=pl-cce>\&#39;</span> &#39;</span>,<span class=pl-s>&#39;.&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L315" class="blob-num js-line-number" data-line-number="315"></td>
        <td id="LC315" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacecaracter</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Height&#39;</span>,<span class=pl-s>&#39;<span class=pl-cce>\&quot;</span>&#39;</span>,<span class=pl-s>&#39;&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L316" class="blob-num js-line-number" data-line-number="316"></td>
        <td id="LC316" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Substitui caracteres da coluna Weight (Peso)</span></td>
      </tr>
      <tr>
        <td id="L317" class="blob-num js-line-number" data-line-number="317"></td>
        <td id="LC317" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacecaracter</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Weight&#39;</span>,<span class=pl-s>&#39; lbs.&#39;</span>,<span class=pl-s>&#39;&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L318" class="blob-num js-line-number" data-line-number="318"></td>
        <td id="LC318" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Substitui caracteres da coluna Reach (Envergadura)</span></td>
      </tr>
      <tr>
        <td id="L319" class="blob-num js-line-number" data-line-number="319"></td>
        <td id="LC319" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacecaracter</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Reach&#39;</span>,<span class=pl-s>&#39;<span class=pl-cce>\&quot;</span>&#39;</span>,<span class=pl-s>&#39;&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L320" class="blob-num js-line-number" data-line-number="320"></td>
        <td id="LC320" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Substitui os valores NaN da coluna Height  (Altura) por -1 (menos um)</span></td>
      </tr>
      <tr>
        <td id="L321" class="blob-num js-line-number" data-line-number="321"></td>
        <td id="LC321" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacenan</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Height&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L322" class="blob-num js-line-number" data-line-number="322"></td>
        <td id="LC322" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Substitui os valores NaN da coluna Weight  (Peso) por -1 (menos um)</span></td>
      </tr>
      <tr>
        <td id="L323" class="blob-num js-line-number" data-line-number="323"></td>
        <td id="LC323" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacenan</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Weight&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L324" class="blob-num js-line-number" data-line-number="324"></td>
        <td id="LC324" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Substitui os valores NaN da coluna Reach (Envergadura) por -1 (menos um)</span></td>
      </tr>
      <tr>
        <td id="L325" class="blob-num js-line-number" data-line-number="325"></td>
        <td id="LC325" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacenan</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L326" class="blob-num js-line-number" data-line-number="326"></td>
        <td id="LC326" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Converte a coluna Height (Altura) de string para float</span></td>
      </tr>
      <tr>
        <td id="L327" class="blob-num js-line-number" data-line-number="327"></td>
        <td id="LC327" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetofloat</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Height&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L328" class="blob-num js-line-number" data-line-number="328"></td>
        <td id="LC328" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Converte a coluna Weight (Peso) de string para inteiro</span></td>
      </tr>
      <tr>
        <td id="L329" class="blob-num js-line-number" data-line-number="329"></td>
        <td id="LC329" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Weight&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L330" class="blob-num js-line-number" data-line-number="330"></td>
        <td id="LC330" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Converte a coluna Reach (Envergadura) de string para inteiro</span></td>
      </tr>
      <tr>
        <td id="L331" class="blob-num js-line-number" data-line-number="331"></td>
        <td id="LC331" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L332" class="blob-num js-line-number" data-line-number="332"></td>
        <td id="LC332" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Converte a coluna Stance (Postura) de object para string</span></td>
      </tr>
      <tr>
        <td id="L333" class="blob-num js-line-number" data-line-number="333"></td>
        <td id="LC333" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetostr</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Stance&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L334" class="blob-num js-line-number" data-line-number="334"></td>
        <td id="LC334" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L335" class="blob-num js-line-number" data-line-number="335"></td>
        <td id="LC335" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### GERA COLUNAS DUMMIES PARA CATEGORIAS DE POSTURA DO LUTADOR</span></td>
      </tr>
      <tr>
        <td id="L336" class="blob-num js-line-number" data-line-number="336"></td>
        <td id="LC336" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 4) Gera no dataframe colunas dummines para categoria de postura do lutador&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L337" class="blob-num js-line-number" data-line-number="337"></td>
        <td id="LC337" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Substitui os valores da coluna Stance (Postura) criando uma nova coluna para cada valor em Stance</span></td>
      </tr>
      <tr>
        <td id="L338" class="blob-num js-line-number" data-line-number="338"></td>
        <td id="LC338" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>pd</span>.<span class=pl-en>concat</span>([<span class=pl-s1>df_2_fighter_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;Stance&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>), <span class=pl-s1>pd</span>.<span class=pl-en>get_dummies</span>(<span class=pl-s1>df_2_fighter_proc</span>[<span class=pl-s>&#39;Stance&#39;</span>],<span class=pl-s1>prefix</span><span class=pl-c1>=</span><span class=pl-s>&#39;Stance&#39;</span>)], <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L339" class="blob-num js-line-number" data-line-number="339"></td>
        <td id="LC339" class="blob-code blob-code-inner js-file-line"><span class=pl-c>## Converte as colunas Stance para inteiro</span></td>
      </tr>
      <tr>
        <td id="L340" class="blob-num js-line-number" data-line-number="340"></td>
        <td id="LC340" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Stance_Open Stance&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L341" class="blob-num js-line-number" data-line-number="341"></td>
        <td id="LC341" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Stance_Orthodox&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L342" class="blob-num js-line-number" data-line-number="342"></td>
        <td id="LC342" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Stance_Sideways&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L343" class="blob-num js-line-number" data-line-number="343"></td>
        <td id="LC343" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Stance_Southpaw&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L344" class="blob-num js-line-number" data-line-number="344"></td>
        <td id="LC344" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>replacetypetoint</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s>&#39;Stance_Switch&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L345" class="blob-num js-line-number" data-line-number="345"></td>
        <td id="LC345" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L346" class="blob-num js-line-number" data-line-number="346"></td>
        <td id="LC346" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### CALCULA A IDADE DO LUTADOR</span></td>
      </tr>
      <tr>
        <td id="L347" class="blob-num js-line-number" data-line-number="347"></td>
        <td id="LC347" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 5) Calcula a idade lutador&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L348" class="blob-num js-line-number" data-line-number="348"></td>
        <td id="LC348" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Calcula a idade do lutador</span></td>
      </tr>
      <tr>
        <td id="L349" class="blob-num js-line-number" data-line-number="349"></td>
        <td id="LC349" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_2_fighter_proc</span> <span class=pl-c1>=</span> <span class=pl-en>agefither</span>(<span class=pl-s1>df_2_fighter_proc</span>, <span class=pl-s>&#39;DOB&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L350" class="blob-num js-line-number" data-line-number="350"></td>
        <td id="LC350" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L351" class="blob-num js-line-number" data-line-number="351"></td>
        <td id="LC351" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### IMPORTAÇÃO DO DATAFRAME COM DETALHES DAS LUTAS (raw_total_fight_data)</span></td>
      </tr>
      <tr>
        <td id="L352" class="blob-num js-line-number" data-line-number="352"></td>
        <td id="LC352" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 6) Importa para dataframe o arquivo com detalhes das lutas&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L353" class="blob-num js-line-number" data-line-number="353"></td>
        <td id="LC353" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Faz a leitura do arquivo CSV</span></td>
      </tr>
      <tr>
        <td id="L354" class="blob-num js-line-number" data-line-number="354"></td>
        <td id="LC354" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_3_total_fight</span> <span class=pl-c1>=</span> <span class=pl-s1>pd</span>.<span class=pl-en>read_csv</span>(<span class=pl-s>&quot;Dataset/raw_total_fight_data.csv&quot;</span>,<span class=pl-s1>sep</span><span class=pl-c1>=</span><span class=pl-s>&#39;;&#39;</span>,<span class=pl-s1>encoding</span><span class=pl-c1>=</span><span class=pl-s>&quot;ISO-8859-1&quot;</span>)</td>
      </tr>
      <tr>
        <td id="L355" class="blob-num js-line-number" data-line-number="355"></td>
        <td id="LC355" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Exibe as informacoes do dataframe</span></td>
      </tr>
      <tr>
        <td id="L356" class="blob-num js-line-number" data-line-number="356"></td>
        <td id="LC356" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#df_3_total_fight.info()</span></td>
      </tr>
      <tr>
        <td id="L357" class="blob-num js-line-number" data-line-number="357"></td>
        <td id="LC357" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Faz uma copia do dataframe original</span></td>
      </tr>
      <tr>
        <td id="L358" class="blob-num js-line-number" data-line-number="358"></td>
        <td id="LC358" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_3_total_fight</span>.<span class=pl-en>copy</span>()</td>
      </tr>
      <tr>
        <td id="L359" class="blob-num js-line-number" data-line-number="359"></td>
        <td id="LC359" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verifica a existencia de dados faltantes</span></td>
      </tr>
      <tr>
        <td id="L360" class="blob-num js-line-number" data-line-number="360"></td>
        <td id="LC360" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#print(&#39;\n&gt;&gt;&gt; Verifica a existencia de dados faltantes em df_4_total_fight_proc_proc:&#39;)</span></td>
      </tr>
      <tr>
        <td id="L361" class="blob-num js-line-number" data-line-number="361"></td>
        <td id="LC361" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#print(df_4_total_fight_proc.isnull().sum())</span></td>
      </tr>
      <tr>
        <td id="L362" class="blob-num js-line-number" data-line-number="362"></td>
        <td id="LC362" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L363" class="blob-num js-line-number" data-line-number="363"></td>
        <td id="LC363" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Exibe a quantidade de lutadores por categoria em ordem decrescente</span></td>
      </tr>
      <tr>
        <td id="L364" class="blob-num js-line-number" data-line-number="364"></td>
        <td id="LC364" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 7) Exibe a quantidade de lutadores por categoria em ordem decrescente:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L365" class="blob-num js-line-number" data-line-number="365"></td>
        <td id="LC365" class="blob-code blob-code-inner js-file-line"><span class=pl-en>countfigherbycategory</span>(<span class=pl-s1>df_4_total_fight_proc</span>)</td>
      </tr>
      <tr>
        <td id="L366" class="blob-num js-line-number" data-line-number="366"></td>
        <td id="LC366" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L367" class="blob-num js-line-number" data-line-number="367"></td>
        <td id="LC367" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### EXCLUSÃO DE COLUNAS DESNECESSÁRIAS</span></td>
      </tr>
      <tr>
        <td id="L368" class="blob-num js-line-number" data-line-number="368"></td>
        <td id="LC368" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 8) Exclui atributos desnecessários ao processo&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L369" class="blob-num js-line-number" data-line-number="369"></td>
        <td id="LC369" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Exclui colunas desnecessárias para a análise</span></td>
      </tr>
      <tr>
        <td id="L370" class="blob-num js-line-number" data-line-number="370"></td>
        <td id="LC370" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_KD&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L371" class="blob-num js-line-number" data-line-number="371"></td>
        <td id="LC371" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_KD&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L372" class="blob-num js-line-number" data-line-number="372"></td>
        <td id="LC372" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_SIG_STR.&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L373" class="blob-num js-line-number" data-line-number="373"></td>
        <td id="LC373" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_SIG_STR.&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L374" class="blob-num js-line-number" data-line-number="374"></td>
        <td id="LC374" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_SIG_STR_pct&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L375" class="blob-num js-line-number" data-line-number="375"></td>
        <td id="LC375" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_SIG_STR_pct&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L376" class="blob-num js-line-number" data-line-number="376"></td>
        <td id="LC376" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_TOTAL_STR.&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L377" class="blob-num js-line-number" data-line-number="377"></td>
        <td id="LC377" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_TOTAL_STR.&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L378" class="blob-num js-line-number" data-line-number="378"></td>
        <td id="LC378" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_TD&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L379" class="blob-num js-line-number" data-line-number="379"></td>
        <td id="LC379" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_TD&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L380" class="blob-num js-line-number" data-line-number="380"></td>
        <td id="LC380" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_TD_pct&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L381" class="blob-num js-line-number" data-line-number="381"></td>
        <td id="LC381" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_TD_pct&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L382" class="blob-num js-line-number" data-line-number="382"></td>
        <td id="LC382" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_SUB_ATT&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L383" class="blob-num js-line-number" data-line-number="383"></td>
        <td id="LC383" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_SUB_ATT&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L384" class="blob-num js-line-number" data-line-number="384"></td>
        <td id="LC384" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_PASS&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L385" class="blob-num js-line-number" data-line-number="385"></td>
        <td id="LC385" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_PASS&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L386" class="blob-num js-line-number" data-line-number="386"></td>
        <td id="LC386" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_REV&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L387" class="blob-num js-line-number" data-line-number="387"></td>
        <td id="LC387" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_REV&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L388" class="blob-num js-line-number" data-line-number="388"></td>
        <td id="LC388" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_HEAD&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L389" class="blob-num js-line-number" data-line-number="389"></td>
        <td id="LC389" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_HEAD&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L390" class="blob-num js-line-number" data-line-number="390"></td>
        <td id="LC390" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_BODY&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L391" class="blob-num js-line-number" data-line-number="391"></td>
        <td id="LC391" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_BODY&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L392" class="blob-num js-line-number" data-line-number="392"></td>
        <td id="LC392" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_LEG&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L393" class="blob-num js-line-number" data-line-number="393"></td>
        <td id="LC393" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_LEG&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L394" class="blob-num js-line-number" data-line-number="394"></td>
        <td id="LC394" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_DISTANCE&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L395" class="blob-num js-line-number" data-line-number="395"></td>
        <td id="LC395" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_DISTANCE&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L396" class="blob-num js-line-number" data-line-number="396"></td>
        <td id="LC396" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_CLINCH&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L397" class="blob-num js-line-number" data-line-number="397"></td>
        <td id="LC397" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_CLINCH&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L398" class="blob-num js-line-number" data-line-number="398"></td>
        <td id="LC398" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;R_GROUND&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L399" class="blob-num js-line-number" data-line-number="399"></td>
        <td id="LC399" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;B_GROUND&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L400" class="blob-num js-line-number" data-line-number="400"></td>
        <td id="LC400" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;win_by&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L401" class="blob-num js-line-number" data-line-number="401"></td>
        <td id="LC401" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;last_round_time&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L402" class="blob-num js-line-number" data-line-number="402"></td>
        <td id="LC402" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;Referee&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L403" class="blob-num js-line-number" data-line-number="403"></td>
        <td id="LC403" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;date&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L404" class="blob-num js-line-number" data-line-number="404"></td>
        <td id="LC404" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;location&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L405" class="blob-num js-line-number" data-line-number="405"></td>
        <td id="LC405" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;Winner&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L406" class="blob-num js-line-number" data-line-number="406"></td>
        <td id="LC406" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Exclui linhas cujo campo de duração total de rounds (Format) tenha valor menor do que 3</span></td>
      </tr>
      <tr>
        <td id="L407" class="blob-num js-line-number" data-line-number="407"></td>
        <td id="LC407" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-en>delrowinvalidround</span>(<span class=pl-s1>df_4_total_fight_proc</span>)</td>
      </tr>
      <tr>
        <td id="L408" class="blob-num js-line-number" data-line-number="408"></td>
        <td id="LC408" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L409" class="blob-num js-line-number" data-line-number="409"></td>
        <td id="LC409" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### MERGE DOS DATAFRAMES</span></td>
      </tr>
      <tr>
        <td id="L410" class="blob-num js-line-number" data-line-number="410"></td>
        <td id="LC410" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 9) Unifica os dataframes de detalhes dos lutadores e das lutas&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L411" class="blob-num js-line-number" data-line-number="411"></td>
        <td id="LC411" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Unifica os dois dataframes</span></td>
      </tr>
      <tr>
        <td id="L412" class="blob-num js-line-number" data-line-number="412"></td>
        <td id="LC412" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-en>mergedataframe</span>(<span class=pl-s1>df_2_fighter_proc</span>,<span class=pl-s1>df_4_total_fight_proc</span>)</td>
      </tr>
      <tr>
        <td id="L413" class="blob-num js-line-number" data-line-number="413"></td>
        <td id="LC413" class="blob-code blob-code-inner js-file-line"><span class=pl-c># Verifica a existencia de dados faltantes</span></td>
      </tr>
      <tr>
        <td id="L414" class="blob-num js-line-number" data-line-number="414"></td>
        <td id="LC414" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#print(&#39;\n&gt;&gt;&gt; Verifica a existencia de dados faltantes em df_4_total_fight_proc:&#39;)</span></td>
      </tr>
      <tr>
        <td id="L415" class="blob-num js-line-number" data-line-number="415"></td>
        <td id="LC415" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#print(df_4_total_fight_proc.isnull().sum())</span></td>
      </tr>
      <tr>
        <td id="L416" class="blob-num js-line-number" data-line-number="416"></td>
        <td id="LC416" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 10) Visualiza a distribuição de dados do atributo R_Height:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L417" class="blob-num js-line-number" data-line-number="417"></td>
        <td id="LC417" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Height&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L418" class="blob-num js-line-number" data-line-number="418"></td>
        <td id="LC418" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 11) Visualiza a distribuição de dados do atributo R_Weight:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L419" class="blob-num js-line-number" data-line-number="419"></td>
        <td id="LC419" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Weight&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L420" class="blob-num js-line-number" data-line-number="420"></td>
        <td id="LC420" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 12) Visualiza a distribuição de dados do atributo R_Reach:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L421" class="blob-num js-line-number" data-line-number="421"></td>
        <td id="LC421" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L422" class="blob-num js-line-number" data-line-number="422"></td>
        <td id="LC422" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 13) Visualiza a distribuição de dados do atributo R_Age:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L423" class="blob-num js-line-number" data-line-number="423"></td>
        <td id="LC423" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L424" class="blob-num js-line-number" data-line-number="424"></td>
        <td id="LC424" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 14) Visualiza a distribuição de dados do atributo B_Height:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L425" class="blob-num js-line-number" data-line-number="425"></td>
        <td id="LC425" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Height&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L426" class="blob-num js-line-number" data-line-number="426"></td>
        <td id="LC426" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 15) Visualiza a distribuição de dados do atributo B_Weight:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L427" class="blob-num js-line-number" data-line-number="427"></td>
        <td id="LC427" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Weight&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L428" class="blob-num js-line-number" data-line-number="428"></td>
        <td id="LC428" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 16) Visualiza a distribuição de dados do atributo B_Reach:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L429" class="blob-num js-line-number" data-line-number="429"></td>
        <td id="LC429" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L430" class="blob-num js-line-number" data-line-number="430"></td>
        <td id="LC430" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 17) Visualiza a distribuição de dados do atributo B_Age:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L431" class="blob-num js-line-number" data-line-number="431"></td>
        <td id="LC431" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L432" class="blob-num js-line-number" data-line-number="432"></td>
        <td id="LC432" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L433" class="blob-num js-line-number" data-line-number="433"></td>
        <td id="LC433" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### CALCULO DAS IDADES MISSING DOS LUTADORES PELA MEDIA DE CADA CATEGORIA DE PESO</span></td>
      </tr>
      <tr>
        <td id="L434" class="blob-num js-line-number" data-line-number="434"></td>
        <td id="LC434" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 18) Calcula as idades missing dos lutadores pela média da categoria de peso&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L435" class="blob-num js-line-number" data-line-number="435"></td>
        <td id="LC435" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Extrai as colunas categoria e idade dos lutadores</span></td>
      </tr>
      <tr>
        <td id="L436" class="blob-num js-line-number" data-line-number="436"></td>
        <td id="LC436" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_5_R_Age_by_Category</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>filter</span>([<span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s>&#39;R_Age&#39;</span>],<span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L437" class="blob-num js-line-number" data-line-number="437"></td>
        <td id="LC437" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Visualiza a frequencia das idades dos lutadores R pela categoria de peso</span></td>
      </tr>
      <tr>
        <td id="L438" class="blob-num js-line-number" data-line-number="438"></td>
        <td id="LC438" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 18.1) Visualiza a frequencia das idades dos lutadores R pela categoria de peso Lightweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L439" class="blob-num js-line-number" data-line-number="439"></td>
        <td id="LC439" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyagebycategory</span>(<span class=pl-s1>df_5_R_Age_by_Category</span>,<span class=pl-s>&#39;R_Age&#39;</span>,<span class=pl-s>&#39;Lightweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L440" class="blob-num js-line-number" data-line-number="440"></td>
        <td id="LC440" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 18.2) Visualiza a frequencia das idades dos lutadores R pela categoria de peso Welterweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L441" class="blob-num js-line-number" data-line-number="441"></td>
        <td id="LC441" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyagebycategory</span>(<span class=pl-s1>df_5_R_Age_by_Category</span>,<span class=pl-s>&#39;R_Age&#39;</span>,<span class=pl-s>&#39;Welterweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L442" class="blob-num js-line-number" data-line-number="442"></td>
        <td id="LC442" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 18.3) Visualiza a frequencia das idades dos lutadores R pela categoria de peso Middleweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L443" class="blob-num js-line-number" data-line-number="443"></td>
        <td id="LC443" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyagebycategory</span>(<span class=pl-s1>df_5_R_Age_by_Category</span>,<span class=pl-s>&#39;R_Age&#39;</span>,<span class=pl-s>&#39;Middleweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L444" class="blob-num js-line-number" data-line-number="444"></td>
        <td id="LC444" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Identifica a media das idades por categoria</span></td>
      </tr>
      <tr>
        <td id="L445" class="blob-num js-line-number" data-line-number="445"></td>
        <td id="LC445" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_5_R_Age_by_Category</span> <span class=pl-c1>=</span> <span class=pl-en>agefithermean</span>(<span class=pl-s1>df_5_R_Age_by_Category</span>,<span class=pl-s>&#39;R_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L446" class="blob-num js-line-number" data-line-number="446"></td>
        <td id="LC446" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Atribui a media da idade por categoria para os valores de idade missing</span></td>
      </tr>
      <tr>
        <td id="L447" class="blob-num js-line-number" data-line-number="447"></td>
        <td id="LC447" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-en>agefithermissing</span>(<span class=pl-s1>df_5_R_Age_by_Category</span>,<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Age&#39;</span>,<span class=pl-s>&#39;R_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L448" class="blob-num js-line-number" data-line-number="448"></td>
        <td id="LC448" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Extrai as colunas categoria e idade dos lutadores</span></td>
      </tr>
      <tr>
        <td id="L449" class="blob-num js-line-number" data-line-number="449"></td>
        <td id="LC449" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_6_B_Age_by_Category</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>filter</span>([<span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s>&#39;B_Age&#39;</span>],<span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L450" class="blob-num js-line-number" data-line-number="450"></td>
        <td id="LC450" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Visualiza a frequencia das idades dos lutadores B pela categoria de peso</span></td>
      </tr>
      <tr>
        <td id="L451" class="blob-num js-line-number" data-line-number="451"></td>
        <td id="LC451" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 18.4) Visualiza a frequencia das idades dos lutadores B pela categoria de peso Lightweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L452" class="blob-num js-line-number" data-line-number="452"></td>
        <td id="LC452" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyagebycategory</span>(<span class=pl-s1>df_6_B_Age_by_Category</span>,<span class=pl-s>&#39;B_Age&#39;</span>,<span class=pl-s>&#39;Lightweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L453" class="blob-num js-line-number" data-line-number="453"></td>
        <td id="LC453" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 18.5) Visualiza a frequencia das idades dos lutadores B pela categoria de peso Welterweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L454" class="blob-num js-line-number" data-line-number="454"></td>
        <td id="LC454" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyagebycategory</span>(<span class=pl-s1>df_6_B_Age_by_Category</span>,<span class=pl-s>&#39;B_Age&#39;</span>,<span class=pl-s>&#39;Welterweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L455" class="blob-num js-line-number" data-line-number="455"></td>
        <td id="LC455" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 18.6) Visualiza a frequencia das idades dos lutadores B pela categoria de peso Middleweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L456" class="blob-num js-line-number" data-line-number="456"></td>
        <td id="LC456" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyagebycategory</span>(<span class=pl-s1>df_6_B_Age_by_Category</span>,<span class=pl-s>&#39;B_Age&#39;</span>,<span class=pl-s>&#39;Middleweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L457" class="blob-num js-line-number" data-line-number="457"></td>
        <td id="LC457" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Identifica a media das idades por categoria</span></td>
      </tr>
      <tr>
        <td id="L458" class="blob-num js-line-number" data-line-number="458"></td>
        <td id="LC458" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_6_B_Age_by_Category</span> <span class=pl-c1>=</span> <span class=pl-en>agefithermean</span>(<span class=pl-s1>df_6_B_Age_by_Category</span>,<span class=pl-s>&#39;B_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L459" class="blob-num js-line-number" data-line-number="459"></td>
        <td id="LC459" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Atribui a media da idade por categoria para os valores de idade missing</span></td>
      </tr>
      <tr>
        <td id="L460" class="blob-num js-line-number" data-line-number="460"></td>
        <td id="LC460" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-en>agefithermissing</span>(<span class=pl-s1>df_6_B_Age_by_Category</span>,<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Age&#39;</span>,<span class=pl-s>&#39;B_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L461" class="blob-num js-line-number" data-line-number="461"></td>
        <td id="LC461" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L462" class="blob-num js-line-number" data-line-number="462"></td>
        <td id="LC462" class="blob-code blob-code-inner js-file-line"><span class=pl-c>##### CALCULO DAS ENVERGADURAS (REACH) MISSING DOS LUTADORES PELA MEDIA DE CADA CATEGORIA DE PESO</span></td>
      </tr>
      <tr>
        <td id="L463" class="blob-num js-line-number" data-line-number="463"></td>
        <td id="LC463" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 19) Calcula as envergaduras missing dos lutadores pela media da categoria de peso&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L464" class="blob-num js-line-number" data-line-number="464"></td>
        <td id="LC464" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Extrai as colunas categoria e envergadura dos lutadores</span></td>
      </tr>
      <tr>
        <td id="L465" class="blob-num js-line-number" data-line-number="465"></td>
        <td id="LC465" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_7_R_Reach_by_Category</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>filter</span>([<span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s>&#39;R_Reach&#39;</span>],<span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L466" class="blob-num js-line-number" data-line-number="466"></td>
        <td id="LC466" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso</span></td>
      </tr>
      <tr>
        <td id="L467" class="blob-num js-line-number" data-line-number="467"></td>
        <td id="LC467" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 19.1) Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso Lightweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L468" class="blob-num js-line-number" data-line-number="468"></td>
        <td id="LC468" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyreachbycategory</span>(<span class=pl-s1>df_7_R_Reach_by_Category</span>,<span class=pl-s>&#39;R_Reach&#39;</span>,<span class=pl-s>&#39;Lightweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L469" class="blob-num js-line-number" data-line-number="469"></td>
        <td id="LC469" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 19.2) Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso Welterweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L470" class="blob-num js-line-number" data-line-number="470"></td>
        <td id="LC470" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyreachbycategory</span>(<span class=pl-s1>df_7_R_Reach_by_Category</span>,<span class=pl-s>&#39;R_Reach&#39;</span>,<span class=pl-s>&#39;Welterweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L471" class="blob-num js-line-number" data-line-number="471"></td>
        <td id="LC471" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 19.3) Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso Middleweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L472" class="blob-num js-line-number" data-line-number="472"></td>
        <td id="LC472" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyreachbycategory</span>(<span class=pl-s1>df_7_R_Reach_by_Category</span>,<span class=pl-s>&#39;R_Reach&#39;</span>,<span class=pl-s>&#39;Middleweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L473" class="blob-num js-line-number" data-line-number="473"></td>
        <td id="LC473" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Identifica a media das envergaduras por categoria</span></td>
      </tr>
      <tr>
        <td id="L474" class="blob-num js-line-number" data-line-number="474"></td>
        <td id="LC474" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_7_R_Reach_by_Category</span> <span class=pl-c1>=</span> <span class=pl-en>reachfithermean</span>(<span class=pl-s1>df_7_R_Reach_by_Category</span>,<span class=pl-s>&#39;R_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L475" class="blob-num js-line-number" data-line-number="475"></td>
        <td id="LC475" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Atribui a media da envergadura por categoria para os valores de envergadura missing</span></td>
      </tr>
      <tr>
        <td id="L476" class="blob-num js-line-number" data-line-number="476"></td>
        <td id="LC476" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-en>reachfithermissing</span>(<span class=pl-s1>df_7_R_Reach_by_Category</span>,<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Reach&#39;</span>,<span class=pl-s>&#39;R_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L477" class="blob-num js-line-number" data-line-number="477"></td>
        <td id="LC477" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Extrai as colunas categoria e envergadura dos lutadores</span></td>
      </tr>
      <tr>
        <td id="L478" class="blob-num js-line-number" data-line-number="478"></td>
        <td id="LC478" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_8_B_Reach_by_Category</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>filter</span>([<span class=pl-s>&#39;Fight_type&#39;</span>,<span class=pl-s>&#39;B_Reach&#39;</span>],<span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L479" class="blob-num js-line-number" data-line-number="479"></td>
        <td id="LC479" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Visualiza a frequencia das envergaduras dos lutadores R pela categoria de peso</span></td>
      </tr>
      <tr>
        <td id="L480" class="blob-num js-line-number" data-line-number="480"></td>
        <td id="LC480" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 19.4) Visualiza a frequencia das envergaduras dos lutadores B pela categoria de peso Lightweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L481" class="blob-num js-line-number" data-line-number="481"></td>
        <td id="LC481" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyreachbycategory</span>(<span class=pl-s1>df_8_B_Reach_by_Category</span>,<span class=pl-s>&#39;B_Reach&#39;</span>,<span class=pl-s>&#39;Lightweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L482" class="blob-num js-line-number" data-line-number="482"></td>
        <td id="LC482" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 19.5) Visualiza a frequencia das envergaduras dos lutadores B pela categoria de peso Welterweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L483" class="blob-num js-line-number" data-line-number="483"></td>
        <td id="LC483" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyreachbycategory</span>(<span class=pl-s1>df_8_B_Reach_by_Category</span>,<span class=pl-s>&#39;B_Reach&#39;</span>,<span class=pl-s>&#39;Welterweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L484" class="blob-num js-line-number" data-line-number="484"></td>
        <td id="LC484" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 19.6) Visualiza a frequencia das envergaduras dos lutadores B pela categoria de peso Middleweight Bout:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L485" class="blob-num js-line-number" data-line-number="485"></td>
        <td id="LC485" class="blob-code blob-code-inner js-file-line"><span class=pl-en>frequencyreachbycategory</span>(<span class=pl-s1>df_8_B_Reach_by_Category</span>,<span class=pl-s>&#39;B_Reach&#39;</span>,<span class=pl-s>&#39;Middleweight Bout&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L486" class="blob-num js-line-number" data-line-number="486"></td>
        <td id="LC486" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Identifica a media das envergaduras por categoria</span></td>
      </tr>
      <tr>
        <td id="L487" class="blob-num js-line-number" data-line-number="487"></td>
        <td id="LC487" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_8_B_Reach_by_Category</span> <span class=pl-c1>=</span> <span class=pl-en>reachfithermean</span>(<span class=pl-s1>df_8_B_Reach_by_Category</span>,<span class=pl-s>&#39;B_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L488" class="blob-num js-line-number" data-line-number="488"></td>
        <td id="LC488" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Atribui a media da envergadura por categoria para os valores de envergadura missing</span></td>
      </tr>
      <tr>
        <td id="L489" class="blob-num js-line-number" data-line-number="489"></td>
        <td id="LC489" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-en>reachfithermissing</span>(<span class=pl-s1>df_8_B_Reach_by_Category</span>,<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Reach&#39;</span>,<span class=pl-s>&#39;B_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L490" class="blob-num js-line-number" data-line-number="490"></td>
        <td id="LC490" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Exclui os valores de envergadura cuja media pela categoria nao foi identificada</span></td>
      </tr>
      <tr>
        <td id="L491" class="blob-num js-line-number" data-line-number="491"></td>
        <td id="LC491" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Reach</span> <span class=pl-c1>!=</span> <span class=pl-c1>0</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Reach</span> <span class=pl-c1>!=</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>)]</td>
      </tr>
      <tr>
        <td id="L492" class="blob-num js-line-number" data-line-number="492"></td>
        <td id="LC492" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Reach</span> <span class=pl-c1>!=</span> <span class=pl-c1>0</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Reach</span> <span class=pl-c1>!=</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>)]</td>
      </tr>
      <tr>
        <td id="L493" class="blob-num js-line-number" data-line-number="493"></td>
        <td id="LC493" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Exclui os valores de idade cuja media pela categoria nao foi identificada</span></td>
      </tr>
      <tr>
        <td id="L494" class="blob-num js-line-number" data-line-number="494"></td>
        <td id="LC494" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Age</span> <span class=pl-c1>!=</span> <span class=pl-c1>0</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Age</span> <span class=pl-c1>!=</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>)]</td>
      </tr>
      <tr>
        <td id="L495" class="blob-num js-line-number" data-line-number="495"></td>
        <td id="LC495" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Exclui os valores de altura não foi identificada</span></td>
      </tr>
      <tr>
        <td id="L496" class="blob-num js-line-number" data-line-number="496"></td>
        <td id="LC496" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Height</span> <span class=pl-c1>!=</span> <span class=pl-c1>0</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Height</span> <span class=pl-c1>!=</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>)]</td>
      </tr>
      <tr>
        <td id="L497" class="blob-num js-line-number" data-line-number="497"></td>
        <td id="LC497" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Height</span> <span class=pl-c1>!=</span> <span class=pl-c1>0</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Height</span> <span class=pl-c1>!=</span> <span class=pl-c1>-</span><span class=pl-c1>1</span>)]</td>
      </tr>
      <tr>
        <td id="L498" class="blob-num js-line-number" data-line-number="498"></td>
        <td id="LC498" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Elimina os outliers do atributo R_Weight</span></td>
      </tr>
      <tr>
        <td id="L499" class="blob-num js-line-number" data-line-number="499"></td>
        <td id="LC499" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Weight</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>245</span>)]</td>
      </tr>
      <tr>
        <td id="L500" class="blob-num js-line-number" data-line-number="500"></td>
        <td id="LC500" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Age</span> <span class=pl-c1>&gt;=</span> <span class=pl-c1>23</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Age</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>50</span>)]</td>
      </tr>
      <tr>
        <td id="L501" class="blob-num js-line-number" data-line-number="501"></td>
        <td id="LC501" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Reach</span> <span class=pl-c1>&gt;=</span> <span class=pl-c1>63</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>R_Reach</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>82</span>)]</td>
      </tr>
      <tr>
        <td id="L502" class="blob-num js-line-number" data-line-number="502"></td>
        <td id="LC502" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Weight</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>245</span>)]</td>
      </tr>
      <tr>
        <td id="L503" class="blob-num js-line-number" data-line-number="503"></td>
        <td id="LC503" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Age</span> <span class=pl-c1>&gt;=</span> <span class=pl-c1>24</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Age</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>48</span>)]</td>
      </tr>
      <tr>
        <td id="L504" class="blob-num js-line-number" data-line-number="504"></td>
        <td id="LC504" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>[(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Reach</span> <span class=pl-c1>&gt;=</span> <span class=pl-c1>64</span>) <span class=pl-c1>&amp;</span> (<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-v>B_Reach</span> <span class=pl-c1>&lt;=</span> <span class=pl-c1>80</span>)]</td>
      </tr>
      <tr>
        <td id="L505" class="blob-num js-line-number" data-line-number="505"></td>
        <td id="LC505" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L506" class="blob-num js-line-number" data-line-number="506"></td>
        <td id="LC506" class="blob-code blob-code-inner js-file-line"><span class=pl-c>###### MACHINE LEARNING</span></td>
      </tr>
      <tr>
        <td id="L507" class="blob-num js-line-number" data-line-number="507"></td>
        <td id="LC507" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Inclui no dataframe o atributo target</span></td>
      </tr>
      <tr>
        <td id="L508" class="blob-num js-line-number" data-line-number="508"></td>
        <td id="LC508" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 20) Incluiu no dataframe o atributo Target&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L509" class="blob-num js-line-number" data-line-number="509"></td>
        <td id="LC509" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span> <span class=pl-c1>=</span> <span class=pl-en>preparedftoml</span>(<span class=pl-s1>df_4_total_fight_proc</span>)</td>
      </tr>
      <tr>
        <td id="L510" class="blob-num js-line-number" data-line-number="510"></td>
        <td id="LC510" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;last_round&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L511" class="blob-num js-line-number" data-line-number="511"></td>
        <td id="LC511" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;Rounds&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>, <span class=pl-s1>inplace</span><span class=pl-c1>=</span><span class=pl-c1>True</span>)</td>
      </tr>
      <tr>
        <td id="L512" class="blob-num js-line-number" data-line-number="512"></td>
        <td id="LC512" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Visualiza a correlação em gráfico para cada lutador</span></td>
      </tr>
      <tr>
        <td id="L513" class="blob-num js-line-number" data-line-number="513"></td>
        <td id="LC513" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 21) Exibe a correlação entre os atributos do lutador R:&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L514" class="blob-num js-line-number" data-line-number="514"></td>
        <td id="LC514" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc_R</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>copy</span>()</td>
      </tr>
      <tr>
        <td id="L515" class="blob-num js-line-number" data-line-number="515"></td>
        <td id="LC515" class="blob-code blob-code-inner js-file-line"><span class=pl-en>figthercorrelation</span> (<span class=pl-s1>df_4_total_fight_proc_R</span>,<span class=pl-s>&#39;B_Height&#39;</span>,<span class=pl-s>&#39;B_Weight&#39;</span>,<span class=pl-s>&#39;B_Reach&#39;</span>,<span class=pl-s>&#39;B_Age&#39;</span>,<span class=pl-s>&#39;B_Stance_Open&#39;</span>,<span class=pl-s>&#39;B_Stance_Orthodox&#39;</span>,<span class=pl-s>&#39;B_Stance_Sideways&#39;</span>,<span class=pl-s>&#39;B_Stance_Southpaw&#39;</span>,<span class=pl-s>&#39;B_Stance_Switch&#39;</span>,<span class=pl-s>&#39;R_Stance_Open&#39;</span>,<span class=pl-s>&#39;R_Stance_Orthodox&#39;</span>,<span class=pl-s>&#39;R_Stance_Sideways&#39;</span>,<span class=pl-s>&#39;R_Stance_Southpaw&#39;</span>,<span class=pl-s>&#39;R_Stance_Switch&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L516" class="blob-num js-line-number" data-line-number="516"></td>
        <td id="LC516" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 22) Exibe a correlação entre os atributos do lutador B:&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L517" class="blob-num js-line-number" data-line-number="517"></td>
        <td id="LC517" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_4_total_fight_proc_B</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>copy</span>()</td>
      </tr>
      <tr>
        <td id="L518" class="blob-num js-line-number" data-line-number="518"></td>
        <td id="LC518" class="blob-code blob-code-inner js-file-line"><span class=pl-en>figthercorrelation</span> (<span class=pl-s1>df_4_total_fight_proc_B</span>,<span class=pl-s>&#39;R_Height&#39;</span>,<span class=pl-s>&#39;R_Weight&#39;</span>,<span class=pl-s>&#39;R_Reach&#39;</span>,<span class=pl-s>&#39;R_Age&#39;</span>,<span class=pl-s>&#39;B_Stance_Open&#39;</span>,<span class=pl-s>&#39;B_Stance_Orthodox&#39;</span>,<span class=pl-s>&#39;B_Stance_Sideways&#39;</span>,<span class=pl-s>&#39;B_Stance_Southpaw&#39;</span>,<span class=pl-s>&#39;B_Stance_Switch&#39;</span>,<span class=pl-s>&#39;R_Stance_Open&#39;</span>,<span class=pl-s>&#39;R_Stance_Orthodox&#39;</span>,<span class=pl-s>&#39;R_Stance_Sideways&#39;</span>,<span class=pl-s>&#39;R_Stance_Southpaw&#39;</span>,<span class=pl-s>&#39;R_Stance_Switch&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L519" class="blob-num js-line-number" data-line-number="519"></td>
        <td id="LC519" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verifica como os dados da varíavel predita estão distribuídos</span></td>
      </tr>
      <tr>
        <td id="L520" class="blob-num js-line-number" data-line-number="520"></td>
        <td id="LC520" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 23) Exibe a  distribuição de  dados da variável predita:&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L521" class="blob-num js-line-number" data-line-number="521"></td>
        <td id="LC521" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>num_true</span> <span class=pl-c1>=</span> <span class=pl-en>len</span>(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>df_4_total_fight_proc</span>[<span class=pl-s>&#39;Target&#39;</span>] <span class=pl-c1>==</span> <span class=pl-c1>1</span>])</td>
      </tr>
      <tr>
        <td id="L522" class="blob-num js-line-number" data-line-number="522"></td>
        <td id="LC522" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>num_false</span> <span class=pl-c1>=</span> <span class=pl-en>len</span>(<span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-s1>loc</span>[<span class=pl-s1>df_4_total_fight_proc</span>[<span class=pl-s>&#39;Target&#39;</span>] <span class=pl-c1>==</span> <span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L523" class="blob-num js-line-number" data-line-number="523"></td>
        <td id="LC523" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;<span class=pl-cce>\n</span>Número de Casos Verdadeiros: {0} ({1:2.2f}%)&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>num_true</span>, (<span class=pl-s1>num_true</span><span class=pl-c1>/</span> (<span class=pl-s1>num_true</span> <span class=pl-c1>+</span> <span class=pl-s1>num_false</span>)) <span class=pl-c1>*</span> <span class=pl-c1>100</span>))</td>
      </tr>
      <tr>
        <td id="L524" class="blob-num js-line-number" data-line-number="524"></td>
        <td id="LC524" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;Número de Casos Falsos     : {0} ({1:2.2f}%)&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>num_false</span>, (<span class=pl-s1>num_false</span><span class=pl-c1>/</span> (<span class=pl-s1>num_true</span> <span class=pl-c1>+</span> <span class=pl-s1>num_false</span>)) <span class=pl-c1>*</span> <span class=pl-c1>100</span>))</td>
      </tr>
      <tr>
        <td id="L525" class="blob-num js-line-number" data-line-number="525"></td>
        <td id="LC525" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#print(&#39;\n&gt;&gt;&gt; Verifica a distribuição de dados dos atributos populados no merge:&#39;)</span></td>
      </tr>
      <tr>
        <td id="L526" class="blob-num js-line-number" data-line-number="526"></td>
        <td id="LC526" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 24) Visualiza a distribuição de dados do atributo R_Height:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L527" class="blob-num js-line-number" data-line-number="527"></td>
        <td id="LC527" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Height&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L528" class="blob-num js-line-number" data-line-number="528"></td>
        <td id="LC528" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 25) Visualiza a distribuição de dados do atributo R_Weight:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L529" class="blob-num js-line-number" data-line-number="529"></td>
        <td id="LC529" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Weight&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L530" class="blob-num js-line-number" data-line-number="530"></td>
        <td id="LC530" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 26) Visualiza a distribuição de dados do atributo R_Reach:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L531" class="blob-num js-line-number" data-line-number="531"></td>
        <td id="LC531" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L532" class="blob-num js-line-number" data-line-number="532"></td>
        <td id="LC532" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 27) Visualiza a distribuição de dados do atributo R_Age:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L533" class="blob-num js-line-number" data-line-number="533"></td>
        <td id="LC533" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;R_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L534" class="blob-num js-line-number" data-line-number="534"></td>
        <td id="LC534" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 28) Visualiza a distribuição de dados do atributo B_Height:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L535" class="blob-num js-line-number" data-line-number="535"></td>
        <td id="LC535" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Height&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L536" class="blob-num js-line-number" data-line-number="536"></td>
        <td id="LC536" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 29) Visualiza a distribuição de dados do atributo B_Weight:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L537" class="blob-num js-line-number" data-line-number="537"></td>
        <td id="LC537" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Weight&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L538" class="blob-num js-line-number" data-line-number="538"></td>
        <td id="LC538" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 30) Visualiza a distribuição de dados do atributo B_Reach:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L539" class="blob-num js-line-number" data-line-number="539"></td>
        <td id="LC539" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Reach&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L540" class="blob-num js-line-number" data-line-number="540"></td>
        <td id="LC540" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 31) Visualiza a distribuição de dados do atributo B_Age:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L541" class="blob-num js-line-number" data-line-number="541"></td>
        <td id="LC541" class="blob-code blob-code-inner js-file-line"><span class=pl-en>boxplotattributes</span>(<span class=pl-s1>df_4_total_fight_proc</span>,<span class=pl-s>&#39;B_Age&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L542" class="blob-num js-line-number" data-line-number="542"></td>
        <td id="LC542" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Prepara dataframe para ML</span></td>
      </tr>
      <tr>
        <td id="L543" class="blob-num js-line-number" data-line-number="543"></td>
        <td id="LC543" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 32) Prepara o dataframe unificado para Machine Learning&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L544" class="blob-num js-line-number" data-line-number="544"></td>
        <td id="LC544" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>df_9_total_fight_ml</span> <span class=pl-c1>=</span> <span class=pl-s1>df_4_total_fight_proc</span>.<span class=pl-en>copy</span>()</td>
      </tr>
      <tr>
        <td id="L545" class="blob-num js-line-number" data-line-number="545"></td>
        <td id="LC545" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Cria modelo de ML</span></td>
      </tr>
      <tr>
        <td id="L546" class="blob-num js-line-number" data-line-number="546"></td>
        <td id="LC546" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 32.1) Separa os dados entre treino e teste&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L547" class="blob-num js-line-number" data-line-number="547"></td>
        <td id="LC547" class="blob-code blob-code-inner js-file-line"><span class=pl-v>X</span> <span class=pl-c1>=</span> <span class=pl-s1>df_9_total_fight_ml</span>.<span class=pl-en>drop</span>(<span class=pl-s>&#39;Target&#39;</span>, <span class=pl-s1>axis</span><span class=pl-c1>=</span><span class=pl-c1>1</span>)</td>
      </tr>
      <tr>
        <td id="L548" class="blob-num js-line-number" data-line-number="548"></td>
        <td id="LC548" class="blob-code blob-code-inner js-file-line"><span class=pl-v>Y</span> <span class=pl-c1>=</span> <span class=pl-s1>df_9_total_fight_ml</span>[<span class=pl-s>&#39;Target&#39;</span>]</td>
      </tr>
      <tr>
        <td id="L549" class="blob-num js-line-number" data-line-number="549"></td>
        <td id="LC549" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Definindo a taxa de split</span></td>
      </tr>
      <tr>
        <td id="L550" class="blob-num js-line-number" data-line-number="550"></td>
        <td id="LC550" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>split_test_size</span> <span class=pl-c1>=</span> <span class=pl-c1>0.3</span></td>
      </tr>
      <tr>
        <td id="L551" class="blob-num js-line-number" data-line-number="551"></td>
        <td id="LC551" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando dados de treino e de teste</span></td>
      </tr>
      <tr>
        <td id="L552" class="blob-num js-line-number" data-line-number="552"></td>
        <td id="LC552" class="blob-code blob-code-inner js-file-line"><span class=pl-v>X_treino</span>, <span class=pl-v>X_teste</span>, <span class=pl-v>Y_treino</span>, <span class=pl-v>Y_teste</span> <span class=pl-c1>=</span> <span class=pl-en>train_test_split</span>(<span class=pl-v>X</span>, <span class=pl-v>Y</span>, <span class=pl-s1>test_size</span> <span class=pl-c1>=</span> <span class=pl-s1>split_test_size</span>)</td>
      </tr>
      <tr>
        <td id="L553" class="blob-num js-line-number" data-line-number="553"></td>
        <td id="LC553" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> 32.2) Aplica os algoritmos de Machine Learning:&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L554" class="blob-num js-line-number" data-line-number="554"></td>
        <td id="LC554" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Utilizando o classificador Naive Bayes</span></td>
      </tr>
      <tr>
        <td id="L555" class="blob-num js-line-number" data-line-number="555"></td>
        <td id="LC555" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span>Modelo 1 - Classificador Naive Bayes:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L556" class="blob-num js-line-number" data-line-number="556"></td>
        <td id="LC556" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando o modelo</span></td>
      </tr>
      <tr>
        <td id="L557" class="blob-num js-line-number" data-line-number="557"></td>
        <td id="LC557" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>modelo_v1</span> <span class=pl-c1>=</span> <span class=pl-v>GaussianNB</span>()</td>
      </tr>
      <tr>
        <td id="L558" class="blob-num js-line-number" data-line-number="558"></td>
        <td id="LC558" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Treinando o modelo</span></td>
      </tr>
      <tr>
        <td id="L559" class="blob-num js-line-number" data-line-number="559"></td>
        <td id="LC559" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>modelo_v1</span>.<span class=pl-en>fit</span>(<span class=pl-v>X_treino</span>, <span class=pl-v>Y_treino</span>.<span class=pl-en>ravel</span>())</td>
      </tr>
      <tr>
        <td id="L560" class="blob-num js-line-number" data-line-number="560"></td>
        <td id="LC560" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verificando a exatidão no modelo nos dados de treino</span></td>
      </tr>
      <tr>
        <td id="L561" class="blob-num js-line-number" data-line-number="561"></td>
        <td id="LC561" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>nb_predict_train</span> <span class=pl-c1>=</span> <span class=pl-s1>modelo_v1</span>.<span class=pl-en>predict</span>(<span class=pl-v>X_treino</span>)</td>
      </tr>
      <tr>
        <td id="L562" class="blob-num js-line-number" data-line-number="562"></td>
        <td id="LC562" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;Exatidão Dados Treino (Accuracy): {0:.4f}&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>metrics</span>.<span class=pl-en>accuracy_score</span>(<span class=pl-v>Y_treino</span>, <span class=pl-s1>nb_predict_train</span>)))</td>
      </tr>
      <tr>
        <td id="L563" class="blob-num js-line-number" data-line-number="563"></td>
        <td id="LC563" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L564" class="blob-num js-line-number" data-line-number="564"></td>
        <td id="LC564" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando uma Confusion Matrix dos dados de treino</span></td>
      </tr>
      <tr>
        <td id="L565" class="blob-num js-line-number" data-line-number="565"></td>
        <td id="LC565" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>matrix</span> <span class=pl-c1>=</span> <span class=pl-s1>metrics</span>.<span class=pl-en>confusion_matrix</span>(<span class=pl-v>Y_treino</span>, <span class=pl-s1>nb_predict_train</span>, <span class=pl-s1>labels</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>,<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L566" class="blob-num js-line-number" data-line-number="566"></td>
        <td id="LC566" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;Confusion Matrix : <span class=pl-cce>\n</span>&#39;</span>, <span class=pl-s1>matrix</span>)</td>
      </tr>
      <tr>
        <td id="L567" class="blob-num js-line-number" data-line-number="567"></td>
        <td id="LC567" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L568" class="blob-num js-line-number" data-line-number="568"></td>
        <td id="LC568" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tn</span>, <span class=pl-s1>fp</span>, <span class=pl-s1>fn</span>, <span class=pl-s1>tp</span> <span class=pl-c1>=</span> <span class=pl-s1>matrix</span>.<span class=pl-en>ravel</span>()</td>
      </tr>
      <tr>
        <td id="L569" class="blob-num js-line-number" data-line-number="569"></td>
        <td id="LC569" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;TP:&#39;</span>, <span class=pl-s1>tp</span>, <span class=pl-s>&#39;FN:&#39;</span>, <span class=pl-s1>fn</span>, <span class=pl-s>&#39;FP:&#39;</span>, <span class=pl-s1>fp</span>, <span class=pl-s>&#39;TN:&#39;</span>, <span class=pl-s1>tn</span>)</td>
      </tr>
      <tr>
        <td id="L570" class="blob-num js-line-number" data-line-number="570"></td>
        <td id="LC570" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verificando a exatidão no modelo nos dados de teste</span></td>
      </tr>
      <tr>
        <td id="L571" class="blob-num js-line-number" data-line-number="571"></td>
        <td id="LC571" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>nb_predict_test</span> <span class=pl-c1>=</span> <span class=pl-s1>modelo_v1</span>.<span class=pl-en>predict</span>(<span class=pl-v>X_teste</span>)</td>
      </tr>
      <tr>
        <td id="L572" class="blob-num js-line-number" data-line-number="572"></td>
        <td id="LC572" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;<span class=pl-cce>\n</span>Exatidão Dados Teste (Accuracy): {0:.4f}&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>metrics</span>.<span class=pl-en>accuracy_score</span>(<span class=pl-v>Y_teste</span>, <span class=pl-s1>nb_predict_test</span>)))</td>
      </tr>
      <tr>
        <td id="L573" class="blob-num js-line-number" data-line-number="573"></td>
        <td id="LC573" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L574" class="blob-num js-line-number" data-line-number="574"></td>
        <td id="LC574" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando uma Confusion Matrix dos dados de teste</span></td>
      </tr>
      <tr>
        <td id="L575" class="blob-num js-line-number" data-line-number="575"></td>
        <td id="LC575" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>matrix</span> <span class=pl-c1>=</span> <span class=pl-s1>metrics</span>.<span class=pl-en>confusion_matrix</span>(<span class=pl-v>Y_teste</span>, <span class=pl-s1>nb_predict_test</span>, <span class=pl-s1>labels</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>,<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L576" class="blob-num js-line-number" data-line-number="576"></td>
        <td id="LC576" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;Confusion Matrix : <span class=pl-cce>\n</span>&#39;</span>, <span class=pl-s1>matrix</span>)</td>
      </tr>
      <tr>
        <td id="L577" class="blob-num js-line-number" data-line-number="577"></td>
        <td id="LC577" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L578" class="blob-num js-line-number" data-line-number="578"></td>
        <td id="LC578" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tn</span>, <span class=pl-s1>fp</span>, <span class=pl-s1>fn</span>, <span class=pl-s1>tp</span> <span class=pl-c1>=</span> <span class=pl-s1>matrix</span>.<span class=pl-en>ravel</span>()</td>
      </tr>
      <tr>
        <td id="L579" class="blob-num js-line-number" data-line-number="579"></td>
        <td id="LC579" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;TP:&#39;</span>, <span class=pl-s1>tp</span>, <span class=pl-s>&#39;FN:&#39;</span>, <span class=pl-s1>fn</span>, <span class=pl-s>&#39;FP:&#39;</span>, <span class=pl-s1>fp</span>, <span class=pl-s>&#39;TN:&#39;</span>, <span class=pl-s1>tn</span>)</td>
      </tr>
      <tr>
        <td id="L580" class="blob-num js-line-number" data-line-number="580"></td>
        <td id="LC580" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Utilizando o modelo RandomForest</span></td>
      </tr>
      <tr>
        <td id="L581" class="blob-num js-line-number" data-line-number="581"></td>
        <td id="LC581" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span>Modelo 2 - RandomForest:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L582" class="blob-num js-line-number" data-line-number="582"></td>
        <td id="LC582" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando o modelo</span></td>
      </tr>
      <tr>
        <td id="L583" class="blob-num js-line-number" data-line-number="583"></td>
        <td id="LC583" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>modelo_v2</span> <span class=pl-c1>=</span> <span class=pl-v>RandomForestClassifier</span>(<span class=pl-s1>n_estimators</span><span class=pl-c1>=</span><span class=pl-c1>1000</span>)</td>
      </tr>
      <tr>
        <td id="L584" class="blob-num js-line-number" data-line-number="584"></td>
        <td id="LC584" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#modelo_v2 = RandomForestClassifier(n_estimators=1000)</span></td>
      </tr>
      <tr>
        <td id="L585" class="blob-num js-line-number" data-line-number="585"></td>
        <td id="LC585" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>modelo_v2</span>.<span class=pl-en>fit</span>(<span class=pl-v>X_treino</span>, <span class=pl-v>Y_treino</span>.<span class=pl-en>ravel</span>())</td>
      </tr>
      <tr>
        <td id="L586" class="blob-num js-line-number" data-line-number="586"></td>
        <td id="LC586" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verificando a exatidão no modelo nos dados de treino</span></td>
      </tr>
      <tr>
        <td id="L587" class="blob-num js-line-number" data-line-number="587"></td>
        <td id="LC587" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>rf_predict_train</span> <span class=pl-c1>=</span> <span class=pl-s1>modelo_v2</span>.<span class=pl-en>predict</span>(<span class=pl-v>X_treino</span>)</td>
      </tr>
      <tr>
        <td id="L588" class="blob-num js-line-number" data-line-number="588"></td>
        <td id="LC588" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;Exatidão dos Dados de Treino (Accuracy): {0:.4f}&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>metrics</span>.<span class=pl-en>accuracy_score</span>(<span class=pl-v>Y_treino</span>, <span class=pl-s1>rf_predict_train</span>)))</td>
      </tr>
      <tr>
        <td id="L589" class="blob-num js-line-number" data-line-number="589"></td>
        <td id="LC589" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L590" class="blob-num js-line-number" data-line-number="590"></td>
        <td id="LC590" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando uma Confusion Matrix dos dados de treino</span></td>
      </tr>
      <tr>
        <td id="L591" class="blob-num js-line-number" data-line-number="591"></td>
        <td id="LC591" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>matrix</span> <span class=pl-c1>=</span> <span class=pl-s1>metrics</span>.<span class=pl-en>confusion_matrix</span>(<span class=pl-v>Y_treino</span>, <span class=pl-s1>rf_predict_train</span>, <span class=pl-s1>labels</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>,<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L592" class="blob-num js-line-number" data-line-number="592"></td>
        <td id="LC592" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;Confusion Matrix : <span class=pl-cce>\n</span>&#39;</span>, <span class=pl-s1>matrix</span>)</td>
      </tr>
      <tr>
        <td id="L593" class="blob-num js-line-number" data-line-number="593"></td>
        <td id="LC593" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L594" class="blob-num js-line-number" data-line-number="594"></td>
        <td id="LC594" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tn</span>, <span class=pl-s1>fp</span>, <span class=pl-s1>fn</span>, <span class=pl-s1>tp</span> <span class=pl-c1>=</span> <span class=pl-s1>matrix</span>.<span class=pl-en>ravel</span>()</td>
      </tr>
      <tr>
        <td id="L595" class="blob-num js-line-number" data-line-number="595"></td>
        <td id="LC595" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;TP:&#39;</span>, <span class=pl-s1>tp</span>, <span class=pl-s>&#39;FN:&#39;</span>, <span class=pl-s1>fn</span>, <span class=pl-s>&#39;FP:&#39;</span>, <span class=pl-s1>fp</span>, <span class=pl-s>&#39;TN:&#39;</span>, <span class=pl-s1>tn</span>)</td>
      </tr>
      <tr>
        <td id="L596" class="blob-num js-line-number" data-line-number="596"></td>
        <td id="LC596" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verificando a exatidão no modelo nos dados de teste</span></td>
      </tr>
      <tr>
        <td id="L597" class="blob-num js-line-number" data-line-number="597"></td>
        <td id="LC597" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>rf_predict_test</span> <span class=pl-c1>=</span> <span class=pl-s1>modelo_v2</span>.<span class=pl-en>predict</span>(<span class=pl-v>X_teste</span>)</td>
      </tr>
      <tr>
        <td id="L598" class="blob-num js-line-number" data-line-number="598"></td>
        <td id="LC598" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;<span class=pl-cce>\n</span>Exatidão dos Dados de Teste (Accuracy): {0:.4f}&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>metrics</span>.<span class=pl-en>accuracy_score</span>(<span class=pl-v>Y_teste</span>, <span class=pl-s1>rf_predict_test</span>)))</td>
      </tr>
      <tr>
        <td id="L599" class="blob-num js-line-number" data-line-number="599"></td>
        <td id="LC599" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L600" class="blob-num js-line-number" data-line-number="600"></td>
        <td id="LC600" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando uma Confusion Matrix dos dados de teste</span></td>
      </tr>
      <tr>
        <td id="L601" class="blob-num js-line-number" data-line-number="601"></td>
        <td id="LC601" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>matrix</span> <span class=pl-c1>=</span> <span class=pl-s1>metrics</span>.<span class=pl-en>confusion_matrix</span>(<span class=pl-v>Y_teste</span>, <span class=pl-s1>rf_predict_test</span>, <span class=pl-s1>labels</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>,<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L602" class="blob-num js-line-number" data-line-number="602"></td>
        <td id="LC602" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;Confusion Matrix : <span class=pl-cce>\n</span>&#39;</span>, <span class=pl-s1>matrix</span>)</td>
      </tr>
      <tr>
        <td id="L603" class="blob-num js-line-number" data-line-number="603"></td>
        <td id="LC603" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L604" class="blob-num js-line-number" data-line-number="604"></td>
        <td id="LC604" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tn</span>, <span class=pl-s1>fp</span>, <span class=pl-s1>fn</span>, <span class=pl-s1>tp</span> <span class=pl-c1>=</span> <span class=pl-s1>matrix</span>.<span class=pl-en>ravel</span>()</td>
      </tr>
      <tr>
        <td id="L605" class="blob-num js-line-number" data-line-number="605"></td>
        <td id="LC605" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;TP:&#39;</span>, <span class=pl-s1>tp</span>, <span class=pl-s>&#39;FN:&#39;</span>, <span class=pl-s1>fn</span>, <span class=pl-s>&#39;FP:&#39;</span>, <span class=pl-s1>fp</span>, <span class=pl-s>&#39;TN:&#39;</span>, <span class=pl-s1>tn</span>)</td>
      </tr>
      <tr>
        <td id="L606" class="blob-num js-line-number" data-line-number="606"></td>
        <td id="LC606" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Utilizando o modelo Regressão Logística</span></td>
      </tr>
      <tr>
        <td id="L607" class="blob-num js-line-number" data-line-number="607"></td>
        <td id="LC607" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span>Modelo 3 - Regressão Logística:<span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L608" class="blob-num js-line-number" data-line-number="608"></td>
        <td id="LC608" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando o modelo</span></td>
      </tr>
      <tr>
        <td id="L609" class="blob-num js-line-number" data-line-number="609"></td>
        <td id="LC609" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>modelo_v3</span> <span class=pl-c1>=</span> <span class=pl-v>LogisticRegression</span>(<span class=pl-s1>solver</span><span class=pl-c1>=</span><span class=pl-s>&#39;lbfgs&#39;</span>, <span class=pl-s1>max_iter</span><span class=pl-c1>=</span><span class=pl-c1>10000</span>)</td>
      </tr>
      <tr>
        <td id="L610" class="blob-num js-line-number" data-line-number="610"></td>
        <td id="LC610" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#modelo_v3 = LogisticRegression(solver=&#39;lbfgs&#39;, max_iter=10000)</span></td>
      </tr>
      <tr>
        <td id="L611" class="blob-num js-line-number" data-line-number="611"></td>
        <td id="LC611" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>modelo_v3</span>.<span class=pl-en>fit</span>(<span class=pl-v>X_treino</span>, <span class=pl-v>Y_treino</span>.<span class=pl-en>ravel</span>())</td>
      </tr>
      <tr>
        <td id="L612" class="blob-num js-line-number" data-line-number="612"></td>
        <td id="LC612" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verificando a exatidão no modelo nos dados de treino</span></td>
      </tr>
      <tr>
        <td id="L613" class="blob-num js-line-number" data-line-number="613"></td>
        <td id="LC613" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>lr_predict_train</span> <span class=pl-c1>=</span> <span class=pl-s1>modelo_v3</span>.<span class=pl-en>predict</span>(<span class=pl-v>X_treino</span>)</td>
      </tr>
      <tr>
        <td id="L614" class="blob-num js-line-number" data-line-number="614"></td>
        <td id="LC614" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;Exatidão dos Dados de Treino (Accuracy): {0:.4f}&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>metrics</span>.<span class=pl-en>accuracy_score</span>(<span class=pl-v>Y_treino</span>, <span class=pl-s1>lr_predict_train</span>)))</td>
      </tr>
      <tr>
        <td id="L615" class="blob-num js-line-number" data-line-number="615"></td>
        <td id="LC615" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L616" class="blob-num js-line-number" data-line-number="616"></td>
        <td id="LC616" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando uma Confusion Matrix dos dados de treino</span></td>
      </tr>
      <tr>
        <td id="L617" class="blob-num js-line-number" data-line-number="617"></td>
        <td id="LC617" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>matrix</span> <span class=pl-c1>=</span> <span class=pl-s1>metrics</span>.<span class=pl-en>confusion_matrix</span>(<span class=pl-v>Y_treino</span>, <span class=pl-s1>lr_predict_train</span>, <span class=pl-s1>labels</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>,<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L618" class="blob-num js-line-number" data-line-number="618"></td>
        <td id="LC618" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;Confusion Matrix : <span class=pl-cce>\n</span>&#39;</span>, <span class=pl-s1>matrix</span>)</td>
      </tr>
      <tr>
        <td id="L619" class="blob-num js-line-number" data-line-number="619"></td>
        <td id="LC619" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L620" class="blob-num js-line-number" data-line-number="620"></td>
        <td id="LC620" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tn</span>, <span class=pl-s1>fp</span>, <span class=pl-s1>fn</span>, <span class=pl-s1>tp</span> <span class=pl-c1>=</span> <span class=pl-s1>matrix</span>.<span class=pl-en>ravel</span>()</td>
      </tr>
      <tr>
        <td id="L621" class="blob-num js-line-number" data-line-number="621"></td>
        <td id="LC621" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;TP:&#39;</span>, <span class=pl-s1>tp</span>, <span class=pl-s>&#39;FN:&#39;</span>, <span class=pl-s1>fn</span>, <span class=pl-s>&#39;FP:&#39;</span>, <span class=pl-s1>fp</span>, <span class=pl-s>&#39;TN:&#39;</span>, <span class=pl-s1>tn</span>)</td>
      </tr>
      <tr>
        <td id="L622" class="blob-num js-line-number" data-line-number="622"></td>
        <td id="LC622" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Verificando a exatidão no modelo nos dados de teste</span></td>
      </tr>
      <tr>
        <td id="L623" class="blob-num js-line-number" data-line-number="623"></td>
        <td id="LC623" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>lr_predict_test</span> <span class=pl-c1>=</span> <span class=pl-s1>modelo_v3</span>.<span class=pl-en>predict</span>(<span class=pl-v>X_teste</span>)</td>
      </tr>
      <tr>
        <td id="L624" class="blob-num js-line-number" data-line-number="624"></td>
        <td id="LC624" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&quot;<span class=pl-cce>\n</span>Exatidão dos Dados de Teste (Accuracy): {0:.4f}&quot;</span>.<span class=pl-en>format</span>(<span class=pl-s1>metrics</span>.<span class=pl-en>accuracy_score</span>(<span class=pl-v>Y_teste</span>, <span class=pl-s1>lr_predict_test</span>)))</td>
      </tr>
      <tr>
        <td id="L625" class="blob-num js-line-number" data-line-number="625"></td>
        <td id="LC625" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L626" class="blob-num js-line-number" data-line-number="626"></td>
        <td id="LC626" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#Criando uma Confusion Matrix dos dados de teste</span></td>
      </tr>
      <tr>
        <td id="L627" class="blob-num js-line-number" data-line-number="627"></td>
        <td id="LC627" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>matrix</span> <span class=pl-c1>=</span> <span class=pl-s1>metrics</span>.<span class=pl-en>confusion_matrix</span>(<span class=pl-v>Y_teste</span>, <span class=pl-s1>lr_predict_test</span>, <span class=pl-s1>labels</span><span class=pl-c1>=</span>[<span class=pl-c1>1</span>,<span class=pl-c1>0</span>])</td>
      </tr>
      <tr>
        <td id="L628" class="blob-num js-line-number" data-line-number="628"></td>
        <td id="LC628" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;Confusion Matrix : <span class=pl-cce>\n</span>&#39;</span>, <span class=pl-s1>matrix</span>)</td>
      </tr>
      <tr>
        <td id="L629" class="blob-num js-line-number" data-line-number="629"></td>
        <td id="LC629" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L630" class="blob-num js-line-number" data-line-number="630"></td>
        <td id="LC630" class="blob-code blob-code-inner js-file-line"><span class=pl-s1>tn</span>, <span class=pl-s1>fp</span>, <span class=pl-s1>fn</span>, <span class=pl-s1>tp</span> <span class=pl-c1>=</span> <span class=pl-s1>matrix</span>.<span class=pl-en>ravel</span>()</td>
      </tr>
      <tr>
        <td id="L631" class="blob-num js-line-number" data-line-number="631"></td>
        <td id="LC631" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;TP:&#39;</span>, <span class=pl-s1>tp</span>, <span class=pl-s>&#39;FN:&#39;</span>, <span class=pl-s1>fn</span>, <span class=pl-s>&#39;FP:&#39;</span>, <span class=pl-s1>fp</span>, <span class=pl-s>&#39;TN:&#39;</span>, <span class=pl-s1>tn</span>)</td>
      </tr>
      <tr>
        <td id="L632" class="blob-num js-line-number" data-line-number="632"></td>
        <td id="LC632" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>()</td>
      </tr>
      <tr>
        <td id="L633" class="blob-num js-line-number" data-line-number="633"></td>
        <td id="LC633" class="blob-code blob-code-inner js-file-line"><span class=pl-en>print</span>(<span class=pl-s>&#39;<span class=pl-cce>\n</span> FIM DO PROCESSO<span class=pl-cce>\n</span><span class=pl-cce>\n</span>&#39;</span>)</td>
      </tr>
      <tr>
        <td id="L634" class="blob-num js-line-number" data-line-number="634"></td>
        <td id="LC634" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#</span></td>
      </tr>
      <tr>
        <td id="L635" class="blob-num js-line-number" data-line-number="635"></td>
        <td id="LC635" class="blob-code blob-code-inner js-file-line"><span class=pl-c>################################# FIM DO PROCESSO #################################</span></td>
      </tr>
      <tr>
        <td id="L636" class="blob-num js-line-number" data-line-number="636"></td>
        <td id="LC636" class="blob-code blob-code-inner js-file-line"><span class=pl-c>#</span></td>
      </tr>
</table>

  <details class="details-reset details-overlay BlobToolbar position-absolute js-file-line-actions dropdown d-none" aria-hidden="true">
    <summary class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1" aria-label="Inline file action toolbar">
      <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M8 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM1.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm13 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"></path></svg>
    </summary>
    <details-menu>
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2" style="width:185px">
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-lines" style="cursor:pointer;">
            Copy lines
          </clipboard-copy>
        </li>
        <li>
          <clipboard-copy role="menuitem" class="dropdown-item" id="js-copy-permalink" style="cursor:pointer;">
            Copy permalink
          </clipboard-copy>
        </li>
        <li><a class="dropdown-item js-update-url-with-hash" id="js-view-git-blame" role="menuitem" href="/itfrancisconeto/PUC_Minas_TCC/blame/d51d8102a054d091c1c0859ea223c1cc04670256/Script_TCC_Francisco_Neto.py">View git blame</a></li>
          <li><a class="dropdown-item" id="js-new-issue" role="menuitem" href="/itfrancisconeto/PUC_Minas_TCC/issues/new">Reference in new issue</a></li>
      </ul>
    </details-menu>
  </details>

  </div>

    </div>

  


  <details class="details-reset details-overlay details-overlay-dark" id="jumpto-line-details-dialog">
    <summary data-hotkey="l" aria-label="Jump to line"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast linejump" aria-label="Jump to line">
      <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form Box-body d-flex" action="" accept-charset="UTF-8" method="get">
        <input class="form-control flex-auto mr-3 linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
        <button type="submit" class="btn" data-close-dialog>Go</button>
</form>    </details-dialog>
  </details>

    <div class="Popover anim-scale-in js-tagsearch-popover"
     hidden
     data-tagsearch-url="/itfrancisconeto/PUC_Minas_TCC/find-definition"
     data-tagsearch-ref="master"
     data-tagsearch-path="Script_TCC_Francisco_Neto.py"
     data-tagsearch-lang="Python"
     data-hydro-click="{&quot;event_type&quot;:&quot;code_navigation.click_on_symbol&quot;,&quot;payload&quot;:{&quot;action&quot;:&quot;click_on_symbol&quot;,&quot;repository_id&quot;:264758279,&quot;ref&quot;:&quot;master&quot;,&quot;language&quot;:&quot;Python&quot;,&quot;originating_url&quot;:&quot;https://github.com/itfrancisconeto/PUC_Minas_TCC/blob/master/Script_TCC_Francisco_Neto.py&quot;,&quot;user_id&quot;:40370984}}"
     data-hydro-click-hmac="9313ae469fea2001b02782c7ba0cf07249934a2e3ad4e677a261bcba0dc732c2">
  <div class="Popover-message Popover-message--large Popover-message--top-left TagsearchPopover mt-1 mb-4 mx-auto Box box-shadow-large">
    <div class="TagsearchPopover-content js-tagsearch-popover-content overflow-auto" style="will-change:transform;">
    </div>
  </div>
</div>




  </div>
</div>

    </main>
  </div>
  

  </div>

        
<div class="footer container-xl width-full p-responsive" role="contentinfo">
  <div class="position-relative d-flex flex-row-reverse flex-lg-row flex-wrap flex-lg-nowrap flex-justify-center flex-lg-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
      <li class="mr-3 mr-lg-0">&copy; 2020 GitHub, Inc.</li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to security, text:security" href="https://github.com/security">Security</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://githubstatus.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://docs.github.com">Help</a></li>

    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon d-none d-lg-block mx-lg-4" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap col-12 col-lg-5 flex-justify-center flex-lg-justify-between mb-2 mb-lg-0">
        <li class="mr-3 mr-lg-0"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.com/pricing" data-ga-click="Footer, go to Pricing, text:Pricing">Pricing</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://docs.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3 mr-lg-0"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
        <li class="mr-3 mr-lg-0"><a href="https://github.blog" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>
    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    <script crossorigin="anonymous" async="async" integrity="sha512-bn/3rKJzBl2H64K38R8KaVcT26vKK7BJQC59lwYc+9fjlHzmy0fwh+hzBtsgTdhIi13dxjzNKWhdSN8WTM9qUw==" type="application/javascript" id="js-conditional-compat" data-src="https://github.githubassets.com/assets/compat-bootstrap-6e7ff7ac.js"></script>
    <script crossorigin="anonymous" integrity="sha512-CxjaMepCmi+z0LTeztU2S8qGD25LyHD6j9t0RSPevy63trFWJVwUM6ipAVLgtpMBBgZ53wq8JPkSeQ6ruaZL2w==" type="application/javascript" src="https://github.githubassets.com/assets/environment-bootstrap-0b18da31.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-fWpoHuxvZSXfbBUpB9xZrMeOn1OBBnQ3RO1Qpyo5hPH/+E/Wrics+UeKW6FaR8Uc6EkKtND0/eKQUZNhWOZG9w==" type="application/javascript" src="https://github.githubassets.com/assets/vendor-7d6a681e.js"></script>
    <script crossorigin="anonymous" async="async" integrity="sha512-ZNZwN+9e4NAzdT0+KmFelwm6CZc8nDMfNKM/PQdTSfyknN/4v+s1Yl6WkFIs32Hy/31PO52krX9IfdH6lodyfg==" type="application/javascript" src="https://github.githubassets.com/assets/frameworks-64d67037.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-/bFYBugcQCjAwrrYqM4EqQe193JblMnS606egBmNbV/iUH7ZQEfnljZbIj34n6SWmayLDvv98mrfdMu81hIAvA==" type="application/javascript" src="https://github.githubassets.com/assets/behaviors-bootstrap-fdb15806.js"></script>
    
      <script crossorigin="anonymous" async="async" integrity="sha512-TjmDUfspgN28WRWfc01tOL0BFS8pI/TAi8TQ665TcA3jG1C3QgfFu0YKa32Z03rlEL8dukbsy+amwBzgGyjESQ==" type="application/javascript" data-module-id="./Sortable.js" data-src="https://github.githubassets.com/assets/Sortable-4e398351.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-IfMEGeu/93qN1/7hW9kHrjLjoz8sU8yHJMOvWmrz6R3Wy7rLelm8F145OPxwF854yGhHt9anAe9BIZMZgLivaA==" type="application/javascript" data-module-id="./drag-drop.js" data-src="https://github.githubassets.com/assets/drag-drop-21f30419.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-iLuC2weaJqL9mYAud2WDWjhd8cJe8dXVxw2KhCH2Rnj6WJvTzlZRmvTtL09wNWX6nRze/TDaQ7gq7BFLchaDYg==" type="application/javascript" data-module-id="./image-crop-element-loader.js" data-src="https://github.githubassets.com/assets/image-crop-element-loader-88bb82db.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-gjvgjSSs1N35phuQcXJQ0945NAuZd2QAJcB2xfLgrd079b9eLXE7KHBDliiNyTG7kG4O3MFDBWlwP9Ye6hBO2g==" type="application/javascript" data-module-id="./jump-to.js" data-src="https://github.githubassets.com/assets/jump-to-823be08d.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-/qu1ZgsxqvzkIPvkOxFe7ioHOms48uHPkflrrH6SHCQ/XNfAydkj2fS7w9yCZR3DDFja9941EMR1+fy4NVBm7g==" type="application/javascript" data-module-id="./profile-pins-element.js" data-src="https://github.githubassets.com/assets/profile-pins-element-feabb566.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-qECv/jhsvLFN77eGNu0cjMR2+zvAlLyhQVTnmayJc5OLZoxMLjQZxZW1hK/dhcYro6Wec/aiF21HYf2N5OilYQ==" type="application/javascript" data-module-id="./randomColor.js" data-src="https://github.githubassets.com/assets/randomColor-a840affe.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-YCODMOpm6dIHRZy9o7UJlwcXMjGIZZzzdwq2JZU6R2VIXZMHJHq/j5KmCFTf+wD+swamrXmse1TPjkdZo4e2RQ==" type="application/javascript" data-module-id="./tweetsodium.js" data-src="https://github.githubassets.com/assets/tweetsodium-60238330.js"></script>
      <script crossorigin="anonymous" async="async" integrity="sha512-PSXN0IjR67dH0v31y2W1LRt8Eb/BaBT+IRe/dy8ffaH6I3zfdAOoMpr80upcJSI0yKceig7dBmBFa+z6rWqKJw==" type="application/javascript" data-module-id="./user-status-submit.js" data-src="https://github.githubassets.com/assets/user-status-submit-3d25cdd0.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-Wme9VMxkOyou2RzQdvG+Ay0KyV0Gf7gM90QSmurM/t/yeold0ko7jyoFfnKQ7nu+NuuoTj7g/Ty4/azctgMeqA==" type="application/javascript" src="https://github.githubassets.com/assets/github-bootstrap-5a67bd54.js"></script>
  <div class="js-stale-session-flash flash flash-warn flash-banner" hidden
    >
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"></path></svg>
    <span class="js-stale-session-flash-signed-in" hidden>You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="js-stale-session-flash-signed-out" hidden>You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <template id="site-details-dialog">
  <details class="details-reset details-overlay details-overlay-dark lh-default text-gray-dark hx_rsm" open>
    <summary role="button" aria-label="Close dialog"></summary>
    <details-dialog class="Box Box--overlay d-flex flex-column anim-fade-in fast hx_rsm-dialog hx_rsm-modal">
      <button class="Box-btn-octicon m-0 btn-octicon position-absolute right-0 top-0" type="button" aria-label="Close dialog" data-close-dialog>
        <svg class="octicon octicon-x" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path></svg>
      </button>
      <div class="octocat-spinner my-6 js-details-dialog-spinner"></div>
    </details-dialog>
  </details>
</template>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>


  </body>
</html>

