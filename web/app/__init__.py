import logging
import secrets

from flask import Flask, render_template, request, session
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import Config


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    if app.config["TRUST_PROXY_HEADERS"]:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    logging.basicConfig(
        level=getattr(logging, app.config["LOG_LEVEL"].upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from .routes import bp
    app.register_blueprint(bp)

    @app.context_processor
    def inject_globals():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return {
            "csrf_token": session["csrf_token"],
            "whatsapp_url": app.config.get("ASPIRATION_WHATSAPP_URL", ""),
        }

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self'; script-src 'self'; form-action 'self'; frame-ancestors 'none'",
        )
        response.headers.setdefault("Permissions-Policy", "camera=(self), geolocation=(), microphone=()")
        return response

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_template(
            "detection.html",
            error=f"Ukuran foto terlalu besar. Maksimum {app.config['MAX_UPLOAD_MB']} MB.",
        ), 413

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error("Unhandled server error: %s", error)
        return render_template("errors/500.html"), 500

    return app
