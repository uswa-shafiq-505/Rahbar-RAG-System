# RAHBAR ui image: nginx serving the static chat page. Build context: ui/
# nginx-unprivileged runs as UID 101 and can listen on 8080; our nginx.conf keeps every
# writable path under /tmp (mounted as an emptyDir in the Deployment).
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 8080
USER 101
CMD ["nginx", "-g", "daemon off;"]
