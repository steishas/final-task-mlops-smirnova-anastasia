terraform {
  required_providers {
    render = {
      source  = "render-oss/render"
      version = "~> 1.0"
    }
  }
}

provider "render" {
  api_key = var.render_api_key
  owner_id = var.render_owner_id
}

resource "render_web_service" "fastapi" {
  name   = "final-task-mlops-mipt"
  plan   = "starter"
  region = "oregon"

  runtime_source = {
    docker = {
      auto_deploy    = true
      branch         = "main"
      dockerfile_path = "./Dockerfile"
      repo_url       = "https://github.com/steishas/final-task-mlops-smirnova-anastasia"
    }
  }

  env_vars = {
    USE_LOCAL_MODEL = {
      value = "true"
    }
  }

  health_check_path = "/health"
}