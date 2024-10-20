import open3d as o3d

mesh = o3d.io.read_triangle_mesh("PIEZAS/3.obj", True)
vis = o3d.visualization.Visualizer()  # Crea un visualizador de Open3D.
vis.create_window(window_name="Open3D", width=1920, height=1080)
vis.add_geometry(mesh)  # Añade el modelo 3D a la ventana de visualización.
vis.get_render_option().load_from_json("render_options.json")  # Carga las opciones de renderización desde un archivo JSON.
vis.get_view_control().set_zoom(0.7)  # Establece el nivel de zoom de la vista.
vis.get_view_control().rotate(300, 1000, xo=0.0, yo=0.0)  # Rota la vista del modelo 3D.
vis.get_view_control().rotate(1000, 0, xo=0.0, yo=0.0)  # Rota la vista del modelo 3D.
vis.poll_events()  # Procesa los eventos de la ventana.
vis.update_renderer()  # Actualiza el renderizador.

vis.poll_events()  # Procesa los eventos de la ventana.
vis.update_renderer()  # Actualiza el renderizador.

vis.run()

# Cierra y destruye la ventana una vez el usuario la cierre
vis.destroy_window()