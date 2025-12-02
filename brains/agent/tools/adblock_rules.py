"""
Ad-Block Rules Module
=====================

Zero-config, built-in ad-blocker for Maven's browser.
Blocks ads, trackers, crypto-miners, and annoyances automatically.

This module is auto-loaded when Maven creates browser contexts, ensuring
clean, fast page loads for web research without any user setup required.

Usage:
    from brains.agent.tools.adblock_rules import apply_adblock
    context = await browser.new_context()
    apply_adblock(context)
"""

from __future__ import annotations
import json
from typing import Any


# EasyList-style blocking rules
# Format: ||domain^ blocks all requests to that domain
ADBLOCK_RULES = """
# Google Ads
||ads.google.com^
||doubleclick.net^
||googlesyndication.com^
||googletagservices.com^
||adservice.google.com^
||pagead2.googlesyndication.com^
||partner.googleadservices.com^
||tpc.googlesyndication.com^
||googleadservices.com^

# Major Ad Networks
||adnxs.com^
||appnexus.com^
||rubiconproject.com^
||pubmatic.com^
||casalemedia.com^
||contextweb.com^
||openx.net^
||smartadserver.com^
||criteo.com^
||criteo.net^
||adform.net^
||adroll.com^
||media.net^

# Yahoo Ads
||adserver.yahoo.com^
||ads.yahoo.com^
||advertising.com^

# Native Ads / Content Recommendation
||outbrain.com^
||taboola.com^
||cdn.taboola.com^
||images.taboola.com^
||trc.taboola.com^
||revcontent.com^
||mgid.com^
||nativeroll.tv^
||sharethrough.com^

# Tracking & Analytics (ad-related)
||adsafeprotected.com^
||scorecardresearch.com^
||krxd.net^
||bluekai.com^
||demdex.net^
||everesttech.net^
||rlcdn.com^
||tapad.com^
||adsrvr.org^
||distiltag.com^
||mc.yandex.ru^

# Pop-ups & Aggressive Ads
||popads.net^
||propellerads.com^
||adblade.com^

# Crypto Miners
||coin-hive.com^
||cryptoloot.com^
||miner.pr0gramm.com^
||webmine.cz^
||coinhive.com^
||crypto-loot.com^

# Amazon Ads
||amazon-adsystem.com^
||ebayadvertising.com^

# Video Ads
||vidoomy.com^
||yieldmo.com^

# Other Ad Networks
||serving-sys.com^
||adition.com^
||ligatus.com^
||tradelab.fr^
||nextmillennium.io^
||smaato.net^
||liqwid.net^
||industrybrains.com^
||cdn.carbonads.com^
||buysellads.net^
||cdn.concert.io^
||bannersnack.com^
||delivery.swid.switchads.com^
"""


def _parse_rules() -> list:
    """Parse ADBLOCK_RULES into a list of domain patterns."""
    rules = []
    for line in ADBLOCK_RULES.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            rules.append(line)
    return rules


def apply_adblock(context: Any) -> None:
    """
    Apply ad-blocking to a Playwright browser context.

    Injects JavaScript that intercepts and blocks network requests
    matching ad domain patterns. Works for XHR, fetch, and dynamic
    element creation (scripts, iframes, images).

    Args:
        context: Playwright BrowserContext object

    Example:
        context = await browser.new_context()
        apply_adblock(context)
    """
    rules = _parse_rules()
    rules_json = json.dumps(rules)

    init_script = f"""
        // Maven Ad-Blocker - Blocks ads, trackers, and annoyances
        (function() {{
            const adblockRules = {rules_json};
            let blockedCount = 0;

            function shouldBlock(url) {{
                if (!url) return false;
                const urlLower = url.toLowerCase();
                return adblockRules.some(rule => {{
                    if (rule.startsWith('||')) {{
                        const domain = rule.slice(2).replace('^', '');
                        return urlLower.includes(domain);
                    }}
                    return urlLower.includes(rule.replace('^', ''));
                }});
            }}

            // Block XMLHttpRequest
            const originalXHROpen = XMLHttpRequest.prototype.open;
            XMLHttpRequest.prototype.open = function(method, url, ...args) {{
                if (shouldBlock(url)) {{
                    blockedCount++;
                    console.log('[MAVEN ADBLOCK] Blocked XHR #' + blockedCount + ':', url.substring(0, 80));
                    this._blocked = true;
                    return;
                }}
                return originalXHROpen.apply(this, [method, url, ...args]);
            }};

            const originalXHRSend = XMLHttpRequest.prototype.send;
            XMLHttpRequest.prototype.send = function(...args) {{
                if (this._blocked) {{
                    Object.defineProperty(this, 'readyState', {{ value: 4 }});
                    Object.defineProperty(this, 'status', {{ value: 0 }});
                    Object.defineProperty(this, 'responseText', {{ value: '' }});
                    if (this.onerror) this.onerror(new Error('Blocked by Maven'));
                    return;
                }}
                return originalXHRSend.apply(this, args);
            }};

            // Block fetch
            const originalFetch = window.fetch;
            window.fetch = function(...args) {{
                const url = typeof args[0] === 'string' ? args[0] : (args[0]?.url || '');
                if (shouldBlock(url)) {{
                    blockedCount++;
                    console.log('[MAVEN ADBLOCK] Blocked fetch #' + blockedCount + ':', url.substring(0, 80));
                    return Promise.reject(new Error('Blocked by Maven'));
                }}
                return originalFetch.apply(this, args);
            }};

            // Block dynamic script/iframe/img src assignment
            function blockSrcProperty(ElementPrototype, propName) {{
                const origDescriptor = Object.getOwnPropertyDescriptor(ElementPrototype, propName);
                if (!origDescriptor) return;

                Object.defineProperty(ElementPrototype, propName, {{
                    get: origDescriptor.get,
                    set: function(value) {{
                        if (shouldBlock(value)) {{
                            blockedCount++;
                            console.log('[MAVEN ADBLOCK] Blocked ' + this.tagName + ' #' + blockedCount + ':', value.substring(0, 80));
                            return;
                        }}
                        if (origDescriptor.set) {{
                            origDescriptor.set.call(this, value);
                        }}
                    }},
                    configurable: true,
                    enumerable: true
                }});
            }}

            // Apply src blocking to relevant elements
            try {{
                if (typeof HTMLScriptElement !== 'undefined') blockSrcProperty(HTMLScriptElement.prototype, 'src');
                if (typeof HTMLIFrameElement !== 'undefined') blockSrcProperty(HTMLIFrameElement.prototype, 'src');
                if (typeof HTMLImageElement !== 'undefined') blockSrcProperty(HTMLImageElement.prototype, 'src');
            }} catch (e) {{
                // Some properties may not be configurable in certain contexts
            }}

            // Expose stats for debugging
            window._mavenAdblockStats = function() {{
                return {{ blocked: blockedCount, rules: adblockRules.length }};
            }};

            console.log('[MAVEN ADBLOCK] Initialized with ' + adblockRules.length + ' rules');
        }})();
    """

    try:
        context.add_init_script(init_script)
        print(f"[ADBLOCK] Applied {len(rules)} ad-blocking rules to browser context")
    except Exception as e:
        print(f"[ADBLOCK] Failed to apply ad-blocking: {e}")


def get_adblock_stats(page: Any) -> dict:
    """
    Get ad-blocking statistics from a page.

    Args:
        page: Playwright Page object

    Returns:
        Dict with 'blocked' count and 'rules' count
    """
    try:
        return page.evaluate("window._mavenAdblockStats ? window._mavenAdblockStats() : {blocked: 0, rules: 0}")
    except Exception:
        return {"blocked": 0, "rules": 0}
