SHELL := /bin/sh

REGISTRY ?= ghcr.io
GITHUB_ORG ?= knows-aggregator
TAG ?= latest

# Add a service here only after services/<name>/Dockerfile is ready to build.
SERVICES := data-preparation

IMAGE_PREFIX := $(REGISTRY)/$(GITHUB_ORG)
SELECTED_SERVICES := $(if $(strip $(CONTAINER)),$(strip $(CONTAINER)),$(SERVICES))

.PHONY: help containers-list containers-validate containers-login containers-build containers-push

help:
	@echo "Container targets:"
	@echo "  make containers-list"
	@echo "  make containers-login GHCR_USER=<github-user>"
	@echo "  make containers-build [CONTAINER=<service>] [TAG=<tag>]"
	@echo "  make containers-push  [CONTAINER=<service>] [TAG=<tag>]"

containers-list:
	@for service in $(SERVICES); do \
		echo "$(IMAGE_PREFIX)/$$service:$(TAG)"; \
	done

containers-validate:
	@if [ -n "$(strip $(CONTAINER))" ]; then \
		case " $(SERVICES) " in \
			*" $(strip $(CONTAINER)) "*) ;; \
			*) echo "Unknown or unpublished service: $(strip $(CONTAINER))" >&2; \
			   echo "Available services: $(SERVICES)" >&2; \
			   exit 2 ;; \
		esac; \
	fi
	@for service in $(SELECTED_SERVICES); do \
		test -f "services/$$service/Dockerfile" || { \
			echo "Missing services/$$service/Dockerfile" >&2; \
			exit 2; \
		}; \
	done

containers-login:
	@test -n "$$GHCR_USER" || { echo "GHCR_USER is required" >&2; exit 2; }
	@test -n "$$CR_PAT" || { echo "CR_PAT is required" >&2; exit 2; }
	@printf '%s' "$$CR_PAT" | docker login "$(REGISTRY)" --username "$$GHCR_USER" --password-stdin

containers-build: containers-validate
	@set -eu; \
	for service in $(SELECTED_SERVICES); do \
		image="$(IMAGE_PREFIX)/$$service:$(TAG)"; \
		echo "Building $$image"; \
		docker build \
			--file "services/$$service/Dockerfile" \
			--tag "$$image" \
			.; \
	done

containers-push: containers-build
	@set -eu; \
	for service in $(SELECTED_SERVICES); do \
		image="$(IMAGE_PREFIX)/$$service:$(TAG)"; \
		echo "Pushing $$image"; \
		docker push "$$image"; \
	done
